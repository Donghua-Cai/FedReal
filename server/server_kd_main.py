# server/server_kd_main.py
import argparse
import concurrent.futures
import logging
import threading
import time
from collections import defaultdict

import grpc
import torch

from proto import fed_pb2, fed_pb2_grpc
from common.config import FedConfig
from common.utils import setup_logger, fmt_bytes
from common.dataset.data_loader import make_global_loaders
from common.dataset.data_transform import get_transform  # 评测用
from server.aggregator_kd import AggregatorKD

class FederatedKDService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_loader, server_test_loader, device: str = "cpu"):
        self.cfg = cfg
        self.logger = logging.getLogger("Server")
        self.agg = AggregatorKD(cfg, public_loader, server_test_loader, device=device)

        # Timer
        self.start_time = None
        self.end_time = None

        # 流量统计
        self._byte_lock = threading.Lock()
        self.bytes_up_logits_total = 0
        self.bytes_down_teacher_total = 0
        self.up_logits_by_client = defaultdict(int)
        self.down_teacher_by_client = defaultdict(int)

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
            model_name=self.cfg.model_name,   # client 的模型名（如 resnet18）
            max_message_mb=self.cfg.max_message_mb,
            server_kd_epochs=self.cfg.server_kd_epochs,
            client_kd_epochs=self.cfg.client_kd_epochs,
            kd_temperature=self.cfg.kd_temperature,
            kd_alpha=self.cfg.kd_alpha,
            num_classes=self.cfg.num_classes,
            public_examples=self.cfg.public_examples,
        )

    # ---- 注册 ----
    def RegisterClient(self, request, context):
        cid, idx = self.agg.register(request.client_name)
        self.logger.info(f"Registered {cid} (index={idx})")
        return fed_pb2.RegisterReply(client_id=cid, client_index=idx, config=self._cfg_to_proto())

    # ---- GetTask（带 phase & 可能带 teacher logits）----
    def GetTask(self, request, context):
        round_id, phase, teacher_bytes = self.agg.get_task(request.client_id)

        # 启动计时（Round0 真正开始：注册满且进入 PHASE_LOCAL_SUP）
        if (self.start_time is None 
            and phase == fed_pb2.PHASE_LOCAL_SUP
            and len(self.agg.registered) >= self.cfg.num_clients):
            self.start_time = time.time()
            self.logger.info("Training timer started.")

        # 仅在首次对该 client 下发 teacher logits 时记录字节
        if teacher_bytes:
            n = len(teacher_bytes)
            self.logger.info(
                f"[Send] teacher_logits bytes={n} ({n/1024/1024:.2f} MB) "
                f"to {request.client_id} for round={round_id}"
            )
            with self._byte_lock:
                self.bytes_down_teacher_total += n
                self.down_teacher_by_client[request.client_id] += n

        return fed_pb2.TaskReply(
            round=round_id,
            phase=phase,
            teacher_logits=teacher_bytes if teacher_bytes else b"",
            config=self._cfg_to_proto(),
        )

    # ---- 客户端上传公共集 logits（本算法核心上行）----
    def UploadPublicLogits(self, request_iterator, context):
        last_client = None
        last_round = None
        total_bytes = 0
        for req in request_iterator:
            last_client = req.client_id
            last_round = req.round
            n = len(req.logits) if req.logits is not None else 0
            self.logger.info(
                f"[Recv] logits bytes={n} ({n/1024/1024:.2f} MB) "
                f"from {req.client_id} for round={req.round}"
            )
            total_bytes += n
            with self._byte_lock:
                self.bytes_up_logits_total += n
                self.up_logits_by_client[req.client_id] += n

            # 转给 aggregator
            self.agg.accept_client_logits(
                client_id=req.client_id,
                round_id=req.round,
                logits_bytes=req.logits,
                num_classes=req.num_classes,
                total_examples=req.total_examples if req.HasField("total_examples") else self.cfg.public_examples,
                local_train_samples=req.local_train_samples if req.HasField("local_train_samples") else 0,
            )
        return fed_pb2.UploadReply(accepted=True, round=self.agg.current_round)

    # ---- 客户端 KD 完成后 ACK（沿用 UploadUpdate）----
    def UploadUpdate(self, request, context):
        ok = self.agg.accept_kd_ack(request.client_id, request.round)
        return fed_pb2.UploadReply(accepted=ok, round=self.agg.current_round)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="0.0.0.0:50052")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--partition_method", type=str, default="dirichlet", choices=["iid","dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client_model", type=str, default="resnet18")  # 写入 config.model_name
    parser.add_argument("--max_message_mb", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # KD 相关
    parser.add_argument("--server_kd_epochs", type=int, default=1)
    parser.add_argument("--client_kd_epochs", type=int, default=1)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--kd_alpha", type=float, default=1.0)
    args = parser.parse_args()

    logger = setup_logger("Server", level=logging.INFO)

    # 全局 Loader（公共无标签 & 服务器测试）
    
    public_unl_loader, server_test_loader = make_global_loaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        seed=args.seed,
        public_ratio=0.1,
        server_test_ratio=0.1,
        train_transform=get_transform(args.dataset_name, "test"),  # 公共集不需要数据增强
        test_transform=get_transform(args.dataset_name, "test"),
    )

    # 推断 num_classes & public_examples
    if server_test_loader is not None:
        base = server_test_loader.dataset
        if isinstance(base, torch.utils.data.Subset):
            num_classes = len(base.dataset.classes)
        else:
            num_classes = len(base.classes)
    else:
        # 退化：从 public loader 推断
        base = public_unl_loader.dataset
        if isinstance(base, torch.utils.data.Subset):
            num_classes = len(base.dataset.classes)
        else:
            num_classes = len(base.classes)
    public_examples = len(public_unl_loader.dataset)

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
        model_name=args.client_model,
        max_message_mb=args.max_message_mb,
    )
    # 补充 KD & dataset 信息
    cfg.server_kd_epochs = args.server_kd_epochs
    cfg.client_kd_epochs = args.client_kd_epochs
    cfg.kd_temperature  = args.kd_temperature
    cfg.kd_alpha        = args.kd_alpha
    cfg.num_classes     = num_classes
    cfg.public_examples = public_examples

    service = FederatedKDService(cfg, public_unl_loader, server_test_loader, device=args.device)

    max_len = cfg.max_message_mb * 1024 * 1024
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )
    add_fn = getattr(fed_pb2_grpc, "add_FederatedServiceServicer_to_server", None)
    if add_fn is None:
        add_fn = fed_pb2_grpc.add_FederatedServiceServicerToServer
    add_fn(service, server)
    server.add_insecure_port(args.bind)
    server.start()
    logger.info(f"Listening on {args.bind}; device={args.device}")

    # 结束时打印 summary
    printed_done = False
    try:
        while True:
            time.sleep(1)
            if service.agg.phase == fed_pb2.PHASE_DONE:
                if not printed_done:
                    if service.start_time is not None:
                        service.end_time = time.time()
                        elapsed = service.end_time - service.start_time
                        logger.info(f"Training completed. Total time: {elapsed:.2f}s ({elapsed/60:.2f} min).")
                    else:
                        logger.info("Training completed.")
                    with service._byte_lock:
                        up_logits  = service.bytes_up_logits_total
                        down_tch   = service.bytes_down_teacher_total
                    logger.info(
                        "[Traffic Summary] "
                        f"up_logits={fmt_bytes(up_logits)}, "
                        f"down_teacher={fmt_bytes(down_tch)}, "
                        f"total={fmt_bytes(up_logits + down_tch)}"
                    )
                    for cid, v in service.up_logits_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_logits={fmt_bytes(v)}")
                    for cid, v in service.down_teacher_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} down_teacher={fmt_bytes(v)}")
                    printed_done = True
                    print("Press Ctrl-C to exit")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)

if __name__ == "__main__":
    main()