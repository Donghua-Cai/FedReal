# server/server_main.py
import argparse
import concurrent.futures
import logging
import os
import time
import threading
from collections import defaultdict

import grpc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from proto import fed_pb2, fed_pb2_grpc  # 由 protoc 生成
from common.config import FedConfig
from common.utils import set_seed, setup_logger, fmt_bytes
from server.aggregator import Aggregator

from common.dataset.data_loader import make_global_loaders
from common.dataset.data_transform import get_transform

class FederatedService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_test_loader, device: str = "cpu"):
        self.cfg = cfg
        self.aggregator = Aggregator(cfg, public_test_loader=public_test_loader, device=device)
        self.logger = logging.getLogger("Server")
        self.start_time = None
        self.end_time = None
        

        self._byte_lock = threading.Lock()
        # 总量
        self.bytes_down_global_total = 0          # 下发全局模型总字节（Server→Clients）
        self.bytes_up_local_total = 0             # 回传本地模型总字节（Clients→Server）
        self.bytes_up_logits_total = 0            # 回传logits总字节（Clients→Server）
        # 分客户端
        self.bytes_down_global_by_client = defaultdict(int)
        self.bytes_up_local_by_client = defaultdict(int)
        self.bytes_up_logits_by_client = defaultdict(int)

    def _cfg_to_proto(self) -> fed_pb2.TrainingConfig:
        return fed_pb2.TrainingConfig(
            num_clients=self.cfg.num_clients,
            total_rounds=self.cfg.total_rounds,
            local_epochs=self.cfg.local_epochs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            partition_method=self.cfg.partition_method,
            dirichlet_alpha=self.cfg.dirichlet_alpha,
            seed=self.cfg.seed,
            sample_fraction=self.cfg.sample_fraction,
            model_name=self.cfg.model_name,
            max_message_mb=self.cfg.max_message_mb,
            num_classes=self.cfg.num_classes
        )

    # ---- RPC: 客户端注册 ----
    def RegisterClient(self, request, context):
        client_id, client_index = self.aggregator.register(request.client_name)
        self.logger.info(f"Registered {client_id} (index={client_index})")
        return fed_pb2.RegisterReply(
            client_id=client_id,
            client_index=client_index,
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 下发训练任务（全局模型+配置）----
    def GetTask(self, request, context):
        round_id, participate, global_bytes = self.aggregator.get_task(request.client_id)

        # 首轮真正开始计时
        if self.start_time is None and round_id < self.cfg.total_rounds and self.aggregator.expected_updates > 0:
            import time as _t
            self.start_time = _t.time()
            self.logger.info("Training timer started.")

        # ✅ 只有在确实要下发（global_bytes 非空）时才打印/统计
        if global_bytes and len(global_bytes) > 0:
            n = len(global_bytes)
            self.logger.info(f"[Send] global_model bytes={n} ({n/1024/1024:.2f} MB) "
                            f"to {request.client_id} for round={round_id}")
            with self._byte_lock:
                self.bytes_down_global_total += n
                self.bytes_down_global_by_client[request.client_id] += n

        # 结束保护：不参与
        if round_id >= self.cfg.total_rounds:
            participate = False

        return fed_pb2.TaskReply(
            round=round_id,
            participate=participate,
            global_model=global_bytes,  # 可能是空字节
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 接收本地更新（模型权重/样本数/指标）----
    def UploadUpdate(self, request, context):
        if request.local_model:
            n = len(request.local_model)
            self.logger.info(f"[Recv] local_model bytes={fmt_bytes(n)} "
                            f"from {request.client_id} for round={request.round}")
            with self._byte_lock:
                self.bytes_up_local_total += n
                self.bytes_up_local_by_client[request.client_id] += n

        ok = self.aggregator.submit_update(
            client_id=request.client_id,
            round_id=request.round,
            local_bytes=request.local_model,
            num_samples=request.num_samples,
        )
        return fed_pb2.UploadReply(accepted=ok, round=self.aggregator.current_round)

    # ---- RPC: 一次性接收客户端上传的公共数据集 logits（非分片）----
    def UploadPublicLogits(self, request_iterator, context):
        """
        流式接收客户端上传的公共数据集 logits。
        现在通常只发 1 个 payload，但流式接口保留扩展性。
        """
        last_round = None
        last_client = None
        total_bytes = 0

        for req in request_iterator:
            last_round = req.round
            last_client = req.client_id

            # 字节统计 & 日志
            n = len(req.logits) if req.logits is not None else 0
            self.logger.info(f"[Recv] logits bytes={n} ({n/1024/1024:.2f} MB) "
                            f"from {req.client_id} for round={req.round}")
            total_bytes += n
            # 可选：写入你在 __init__ 里加的全程统计计数器
            with self._byte_lock:
                self.bytes_up_logits_total += n
                self.bytes_up_logits_by_client[req.client_id] += n

            # ✅ 新增：把 local_train_samples 也传下去（可能没有）
            local_train_samples = req.local_train_samples if req.HasField("local_train_samples") else None
            total_examples = req.total_examples if req.HasField("total_examples") else None

            self.aggregator.accept_public_logits_payload(
                client_id=req.client_id,
                round_id=req.round,
                logits_bytes=req.logits,
                indices=list(req.indices),
                num_classes=int(req.num_classes),
                total_examples=int(total_examples) if total_examples is not None else None,
                local_train_samples=int(local_train_samples) if local_train_samples is not None else None,
            )

        # 流结束，返回 Ack（你现在返回 UploadReply）
        return fed_pb2.UploadReply(accepted=True, round=self.aggregator.current_round)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="0.0.0.0:50051")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="cifar10",
                        help="Dataset name under data/, used to build public test loader")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--partition_method", type=str, default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--max_message_mb", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers (None = auto)")
    args = parser.parse_args()

    set_seed(args.seed)
    # 统一初始化命名 logger
    logger = setup_logger("Server", level=logging.INFO)

    cfg = FedConfig(
        num_clients=args.num_clients,
        total_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        sample_fraction=args.sample_fraction,
        model_name=args.model_name,
        max_message_mb=args.max_message_mb,
    )

    public_unl_loader, server_test_loader = make_global_loaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=cfg.batch_size,
        seed=cfg.seed,                 # 确保与 client 一致
        public_ratio=0.1,
        server_test_ratio=0.1,
        train_transform=get_transform(args.dataset_name, "test"),  # 或者你自己的
        test_transform=get_transform(args.dataset_name, "test"),
    )

    if server_test_loader is not None:
        base_dataset = server_test_loader.dataset
        if isinstance(base_dataset, torch.utils.data.Subset):
            num_classes = len(base_dataset.dataset.classes)
        else:
            num_classes = len(base_dataset.classes)
    elif public_unl_loader is not None:
        base_dataset = public_unl_loader.dataset
        if isinstance(base_dataset, torch.utils.data.Subset):
            num_classes = len(base_dataset.dataset.classes)
        else:
            num_classes = len(base_dataset.classes)
    else:
        num_classes = 10  # fallback，避免报错

    cfg.num_classes = num_classes

    service = FederatedService(cfg, public_test_loader=server_test_loader, device=args.device)

    max_len = cfg.max_message_mb * 1024 * 1024
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )

    # 兼容不同 grpcio-tools 生成的函数名
    add_fn = getattr(fed_pb2_grpc, "add_FederatedServiceServicer_to_server", None)
    if add_fn is None:
        add_fn = fed_pb2_grpc.add_FederatedServiceServicerToServer
    add_fn(service, server)

    server.add_insecure_port(args.bind)
    server.start()
    logger.info(f"Current setting:")
    logger.info(f" - dataset : {args.dataset_name}")
    logger.info(f" - partition : {args.partition_method}")
    logger.info(f" - dirichlet alpha : {args.dirichlet_alpha}")
    logger.info(f" - clients number : {args.num_clients}")
    logger.info(f" - local epochs : {args.local_epochs}")
    logger.info(f"Listening on {args.bind}; device={args.device}")

    # 只打印一次“完成”
    printed_done = False
    try:
        while True:
            time.sleep(1)
            if (service.aggregator.current_round >= cfg.total_rounds
                and service.aggregator.expected_updates == 0
                and not service.aggregator.selected_this_round):
                if not printed_done:
                    if service.start_time is not None:
                        service.end_time = time.time()
                        elapsed = service.end_time - service.start_time
                        logger.info(f"Training completed. Total time: {elapsed:.2f}s ({elapsed/60:.2f} min).")
                    else:
                        logger.info("Training completed.")
                    
                    with service._byte_lock:
                        down = service.bytes_down_global_total
                        up_local = service.bytes_up_local_total
                        up_logits = service.bytes_up_logits_total
                    logger.info(
                        "[Traffic Summary] "
                        f"down_global={fmt_bytes(down)}, "
                        f"up_local={fmt_bytes(up_local)}, "
                        f"up_logits={fmt_bytes(up_logits)}, "
                        f"total={fmt_bytes(down + up_local + up_logits)}"
                    )
                    # 如需查看分客户端明细：
                    for cid, v in service.bytes_down_global_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} down_global={fmt_bytes(v)}")
                    for cid, v in service.bytes_up_local_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_local={fmt_bytes(v)}")
                    for cid, v in service.bytes_up_logits_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_logits={fmt_bytes(v)}")
                    printed_done = True
                    logger.info(f"Server total acc : {service.aggregator.server_eval_acc}")
                    logger.info(f"Server total loss : {service.aggregator.server_eval_loss}")
                    print("Press Ctrl-C to exit")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == "__main__":
    main()