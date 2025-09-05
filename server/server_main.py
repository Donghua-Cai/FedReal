# server/server_main.py
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
from common.utils import set_seed, setup_logger, fmt_bytes
from common.dataset.data_loader import make_global_loaders
from common.dataset.data_transform import get_transform
from server.aggregator import Aggregator


class FederatedService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_unl_loader, server_test_loader, public_labels, device: str = "cpu"):
        self.cfg = cfg
        self.logger = logging.getLogger("Server")
        self.agg = Aggregator(cfg, public_unl_loader=public_unl_loader, server_test_loader=server_test_loader, public_labels=public_labels, device=device)

        # 计时
        self.start_time = None
        self.end_time = None

        # 流量统计
        self._byte_lock = threading.Lock()
        self.bytes_down_model_total = 0
        self.bytes_up_local_total = 0
        self.bytes_up_logits_total = 0
        self.bytes_down_logits_total = 0

        self.bytes_down_model_by_client = defaultdict(int)
        self.bytes_up_local_by_client = defaultdict(int)
        self.bytes_up_logits_by_client = defaultdict(int)
        self.bytes_down_logits_by_client = defaultdict(int)

    # ---- 配置下发 ----
    def _cfg_to_proto(self) -> fed_pb2.TrainingConfig:
        # 若 proto 未含 kd/server 字段，可删去相应行或改用默认
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
            num_classes=self.cfg.num_classes,
            # 供 client KD 读取（如果你的 proto 未包含，请从 client 侧用默认）
            server_kd_epochs=getattr(self.cfg, "server_kd_epochs", 3),
            client_kd_epochs=getattr(self.cfg, "client_kd_epochs", 3),
            kd_temperature=getattr(self.cfg, "kd_temperature", 1.0),
            kd_alpha=getattr(self.cfg, "kd_alpha", 0.5),
            public_examples=self.cfg.public_examples,
        )

    # ---- RPC: 注册 ----
    def RegisterClient(self, request, context):
        cid, idx, gid = self.agg.register(request.client_name)
        self.logger.info(f"Registered {cid} (index={idx}, group={gid})")
        return fed_pb2.RegisterReply(
            client_id=cid,
            client_index=idx,
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 获取任务（含 Phase / 可能的模型或 logits 下发）----
    def GetTask(self, request, context):
        round_id, phase, participate, model_bytes, server_logits_bytes = self.agg.get_task(request.client_id)

        # 训练开始计时：第一轮真正进入 LOCAL_TRAIN 时
        if (self.start_time is None
            and phase == fed_pb2.PHASE_LOCAL_TRAIN
            and self.agg.ready_to_start()):
            self.start_time = time.time()
            self.logger.info("Training timer started.")

        # 下发模型计数
        if model_bytes:
            n = len(model_bytes)
            with self._byte_lock:
                self.bytes_down_model_total += n
                self.bytes_down_model_by_client[request.client_id] += n

        # 下发 server logits 计数
        if server_logits_bytes:
            n = len(server_logits_bytes)
            with self._byte_lock:
                self.bytes_down_logits_total += n
                self.bytes_down_logits_by_client[request.client_id] += n

        return fed_pb2.TaskReply(
            round=round_id,
            phase=phase,
            participate=participate,
            group_model=model_bytes if model_bytes else b"",
            server_logits=server_logits_bytes if server_logits_bytes else b"",
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 客户端上传本地小模型（带 group_id）----
    def UploadUpdate(self, request, context):
        # 客户端 KD 阶段的 ACK（空模型）
        if not request.local_model and self.agg.phase == fed_pb2.PHASE_CLIENT_KD:
            ok = self.agg.submit_client_kd_ack(request.client_id, request.round)
            return fed_pb2.UploadReply(accepted=ok, round=self.agg.current_round)

        # 只有“会被接受”的本地更新才打印日志和计流量
        will_take = self.agg.will_accept_update(request.client_id, request.round)
        if request.local_model and will_take:
            n = len(request.local_model)
            self.logger.info(
                f"[Recv] local_model {fmt_bytes(n)} from {request.client_id} (group={request.group_id}) round={request.round}"
            )
            with self._byte_lock:
                self.bytes_up_local_total += n
                self.bytes_up_local_by_client[request.client_id] += n

        ok = self.agg.submit_update(
            client_id=request.client_id,
            group_id=request.group_id,
            round_id=request.round,
            local_bytes=request.local_model,
            num_samples=request.num_samples,
        )
        return fed_pb2.UploadReply(accepted=ok, round=self.agg.current_round)

    # ---- RPC: 组内代表上传 logits（沿用 UploadPublicLogits）----
    def UploadGroupLogits(self, request, context):
        # 计流量 + 日志
        n = len(request.logits) if request.logits is not None else 0
        self.logger.info(f"[Recv] group_logits {fmt_bytes(n)} from {request.client_id} (group={request.group_id}) round={request.round}")
        with self._byte_lock:
            self.bytes_up_logits_total += n
            self.bytes_up_logits_by_client[request.client_id] += n

        # 转交 Aggregator
        self.agg.accept_group_logits(
            client_id=request.client_id,
            round_id=request.round,
            logits_bytes=request.logits,
            num_classes=request.num_classes,
            total_examples=request.total_examples if request.total_examples > 0 else self.cfg.public_examples,
        )
        return fed_pb2.UploadReply(accepted=True, round=self.agg.current_round)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="0.0.0.0:50051")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--partition_method", type=str, default="iid", choices=["iid", "dirichlet", "shards"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="resnet18")      # client 小模型
    parser.add_argument("--server_model", type=str, default="resnet50")    # server 大模型
    parser.add_argument("--max_message_mb", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # KD
    parser.add_argument("--server_kd_epochs", type=int, default=3)
    parser.add_argument("--client_kd_epochs", type=int, default=3)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    # 分组
    parser.add_argument("--group_num", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger("Server", level=logging.INFO)

    # 公共无标签 + 服务器评测（可选）
    public_unl_loader, server_test_loader, public_labels = make_global_loaders(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        seed=args.seed,
        public_ratio=0.1,
        server_test_ratio=0.1,
        train_transform=get_transform(args.dataset_name, "test"),
        test_transform=get_transform(args.dataset_name, "test"),
        return_index_in_public=True
    )

    # 推断 num_classes/public_examples
    if server_test_loader is not None:
        base = server_test_loader.dataset
        num_classes = len(base.dataset.classes) if isinstance(base, torch.utils.data.Subset) else len(base.classes)
    else:
        base = public_unl_loader.dataset
        num_classes = len(base.dataset.classes) if isinstance(base, torch.utils.data.Subset) else len(base.classes)
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
        model_name=args.model_name,
        max_message_mb=args.max_message_mb,
    )
    # 扩展字段（FedConfig 里加入这些属性即可）
    cfg.num_classes = num_classes
    cfg.public_examples = public_examples
    cfg.group_num = args.group_num
    cfg.server_model_name = args.server_model
    cfg.server_kd_epochs = args.server_kd_epochs
    cfg.client_kd_epochs = args.client_kd_epochs
    cfg.kd_temperature = args.kd_temperature
    cfg.kd_alpha = args.kd_alpha

    service = FederatedService(cfg, public_unl_loader, server_test_loader, public_labels, device=args.device)

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
    logger.info(f"dataset={args.dataset_name}, groups={args.group_num}, clients={args.num_clients}")

    printed_done = False
    try:
        while True:
            time.sleep(1)
            if service.agg.is_all_done():
                if not printed_done:
                    if service.start_time is not None:
                        service.end_time = time.time()
                        elapsed = service.end_time - service.start_time
                        logger.info(f"Training completed. Total time: {elapsed:.2f}s ({elapsed/60:.2f} min).")
                    else:
                        logger.info("Training completed.")
                    with service._byte_lock:
                        down_m = service.bytes_down_model_total
                        up_m = service.bytes_up_local_total
                        up_l = service.bytes_up_logits_total
                        down_l = service.bytes_down_logits_total
                    logger.info(
                        "[Traffic Summary] "
                        f"down_model={fmt_bytes(down_m)}, up_model={fmt_bytes(up_m)}, "
                        f"up_logits={fmt_bytes(up_l)}, down_logits={fmt_bytes(down_l)}, "
                        f"total={fmt_bytes(down_m+up_m+up_l+down_l)}"
                    )
                    printed_done = True
                    logger.info(f"Server total acc : {service.agg.server_eval_acc}")
                    logger.info(f"Server total loss: {service.agg.server_eval_loss}")
                    print("Press Ctrl-C to exit")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == "__main__":
    main()