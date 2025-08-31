import argparse
import concurrent.futures
import grpc
import os
import time
import torch

from proto import fed_pb2, fed_pb2_grpc  # 由 protoc 生成
from common.config import FedConfig
from common.dataset.dataset import build_cifar10_loaders_for_client
from common.serialization import state_dict_to_bytes
from server.aggregator import Aggregator


class FederatedService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_test_loader, device="cpu"):
        self.cfg = cfg
        self.aggregator = Aggregator(cfg, public_test_loader=public_test_loader, device=device)

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

    def RegisterClient(self, request, context):
        client_id, client_index = self.aggregator.register(request.client_name)
        print(f"[Server] Registered {client_id} (index={client_index})")
        return fed_pb2.RegisterReply(
            client_id=client_id,
            client_index=client_index,
            config=self._cfg_to_proto(),
        )

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

    def UploadUpdate(self, request, context):
        ok = self.aggregator.submit_update(
            client_id=request.client_id,
            round_id=request.round,
            local_bytes=request.local_model,
            num_samples=request.num_samples,
        )
        return fed_pb2.UploadReply(accepted=ok, round=self.aggregator.current_round)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="0.0.0.0:50051")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--partition_method", type=str, default="iid", choices=["iid", "dirichlet"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--max_message_mb", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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

    # 为了复用客户端的数据管线，这里使用 client builder 拿到公共测试集
    # 任取 client_index=0 仅为了获得 public_test_loader
    _, _, public_test_loader, _ = build_cifar10_loaders_for_client(
        data_root=args.data_root,
        client_index=0,
        num_clients=cfg.num_clients,
        partition_method=cfg.partition_method,
        dirichlet_alpha=cfg.dirichlet_alpha,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
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
    add_fn = getattr(fed_pb2_grpc, "add_FederatedServiceServicer_to_server", None)
    if add_fn is None:
        add_fn = fed_pb2_grpc.add_FederatedServiceServicerToServer
    add_fn(service, server)
    server.add_insecure_port(args.bind)
    server.start()
    print(f"[Server] Listening on {args.bind}; device={args.device}")

    try:
        while True:
            time.sleep(5)
            if service.aggregator.current_round >= cfg.total_rounds:
                print("[Server] Training completed. Press Ctrl+C to stop.")
                # 可根据需要自动退出
                # break
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == "__main__":
    main()
