# server/server_main.py
import argparse
import concurrent.futures
import logging
import os
import time

import grpc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from proto import fed_pb2, fed_pb2_grpc  # 由 protoc 生成
from common.config import FedConfig
from common.utils import setup_logger
from server.aggregator import Aggregator

from common.dataset.data_transform import test_tf_cifar10


def _make_public_test_loader(data_root: str, dataset_name: str, batch_size: int, num_workers=None):
    """
    从 data/<dataset_name>/test 或 val 构建公共测试集 DataLoader。
    如果两个目录都不存在，则返回 None。
    """
    log = logging.getLogger("Server")

    test_dir = None
    for candidate in ["test", "val"]:
        cand_dir = os.path.join(data_root, dataset_name, candidate)
        if os.path.isdir(cand_dir):
            test_dir = cand_dir
            break

    if test_dir is None:
        log.warning(
            f"No public test dir found under {os.path.join(data_root, dataset_name)} "
            f"(global eval disabled)"
        )
        return None

    has_cuda = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 2 if has_cuda else 0
    pin_memory = True if has_cuda else False

    public_set = datasets.ImageFolder(test_dir, transform=test_tf_cifar10)
    public_loader = DataLoader(
        public_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    log.info(f"Using public test dir: {test_dir} (samples={len(public_set)})")
    return public_loader


class FederatedService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_test_loader, device: str = "cpu"):
        self.cfg = cfg
        self.aggregator = Aggregator(cfg, public_test_loader=public_test_loader, device=device)
        self.logger = logging.getLogger("Server")

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
        # 训练尚未开始或已结束时的简单处理
        if round_id >= self.cfg.total_rounds:
            participate = False
        return fed_pb2.TaskReply(
            round=round_id,
            participate=participate,
            global_model=global_bytes,
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 接收本地更新（模型权重/样本数/指标）----
    def UploadUpdate(self, request, context):
        ok = self.aggregator.submit_update(
            client_id=request.client_id,
            round_id=request.round,
            local_bytes=request.local_model,
            num_samples=request.num_samples,
        )
        return fed_pb2.UploadReply(accepted=ok, round=self.aggregator.current_round)

    # ---- RPC: 一次性接收客户端上传的公共数据集 logits（非分片）----
    def UploadPublicLogits(self, request, context):
        self.aggregator.accept_public_logits_payload(
            client_id=request.client_id,
            round_id=request.round,
            logits_bytes=request.logits,
            indices=list(request.indices),
            num_classes=int(request.num_classes),
            total_examples=int(request.total_examples) if request.HasField("total_examples") else None,
        )
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

    # 使用 ImageFolder(test|val) 构建公共测试集（若不存在则 None）
    public_test_loader = _make_public_test_loader(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        batch_size=cfg.batch_size,
        num_workers=args.num_workers,
    )

    service = FederatedService(cfg, public_test_loader=public_test_loader, device=args.device)

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
                    logger.info("Training completed. Press Ctrl+C to stop.")
                    printed_done = True
                # 如需自动退出，解除下一行注释
                # break
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == "__main__":
    main()