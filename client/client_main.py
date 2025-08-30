import argparse
import time
import grpc
import torch

from proto import fed_pb2, fed_pb2_grpc
from common.dataset.dataset import build_cifar10_loaders_for_client
from common.model.model import create_model
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from client.trainer import train_local, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:50051")
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # 初次连接，使用默认 128MB；后续按服务端回传的配置调整
    max_len = 128 * 1024 * 1024
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )
    stub = fed_pb2_grpc.FederatedServiceStub(channel)

    # 注册，获得 client_id / client_index / 配置
    reg = stub.RegisterClient(fed_pb2.RegisterRequest(client_name=args.client_name))
    client_id = reg.client_id
    client_index = reg.client_index
    cfg = reg.config

    print(f"[Client {client_id}] index={client_index}; device={args.device}")

    # 依据配置构建本地数据加载器（确定性划分）
    train_loader, test_loader, public_test_loader, n_train = build_cifar10_loaders_for_client(
        data_root=args.data_root,
        client_index=client_index,
        num_clients=cfg.num_clients,
        partition_method=cfg.partition_method,
        dirichlet_alpha=cfg.dirichlet_alpha,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    device = torch.device(args.device)

    # 主循环
    last_round = -1
    while True:
        task = stub.GetTask(fed_pb2.GetTaskRequest(client_id=client_id))
        if task.round >= cfg.total_rounds:
            print(f"[Client {client_id}] All rounds finished.")
            break

        if task.round != last_round:
            print(f"[Client {client_id}] Enter round {task.round}")
            last_round = task.round

        if not task.participate:
            time.sleep(1.0)
            continue

        # 拉取并加载全局模型
        model = create_model(task.config.model_name, num_classes=10).to(device)
        state_dict = bytes_to_state_dict(task.global_model)
        model.load_state_dict(state_dict)

        # 本地训练
        optimizer = torch.optim.SGD(model.parameters(), lr=task.config.lr, momentum=task.config.momentum, weight_decay=5e-4)
        train_loss, train_acc = train_local(model, train_loader, epochs=task.config.local_epochs, optimizer=optimizer, device=device)
        test_loss, test_acc = evaluate(model, test_loader, device=device)
        print(f"[Client {client_id}][Round {task.round}] train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        # 回传完整模型（简单实现）
        local_bytes = state_dict_to_bytes(model.state_dict())
        reply = stub.UploadUpdate(
            fed_pb2.UploadRequest(
                client_id=client_id,
                round=task.round,
                local_model=local_bytes,
                num_samples=n_train,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
            )
        )

        if reply.accepted:
            # 继续轮询下一轮
            time.sleep(0.2)
        else:
            # 如果被拒绝（比如轮次错位），稍后重试
            time.sleep(1.0)


if __name__ == "__main__":
    main()

