import argparse
import time
import grpc
import torch
import logging
import collections


from proto import fed_pb2, fed_pb2_grpc
#from common.dataset import build_imagefolder_loaders_for_client
#from common.dataset.data_transform import get_transform
from common.data_utils.data_loader import client_data_loader
from common.model.models_fedext import FedEXTModel
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.utils import setup_logger, set_seed
from client.trainer import train_local, evaluate


def main():
    
    start_time = None
    end_time = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:50051")
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_name", type=str, default="Cifar10", choices=["Cifar10", "Cifar100", "NWPU-RESISC45", "DOTA"],
                        help="Dataset name under data/, e.g., cifar10, nwpu, dota")
    parser.add_argument("--num_classes", type=int)                        
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Dataloader workers (None = auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder_ratio", type=float, default=1.0)
    
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger(f"Client-{args.client_name}", level=logging.INFO)

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
    group_index = reg.group_index
    cfg = reg.config

    logger.info(f"[Client {client_id}] index={client_index}; device={args.device}")

    
    train_loader, test_loader = client_data_loader(args.data_root, args.dataset_name, client_index, args.batch_size)

    # logger.info(f"[Client {client_id}] group_id={group_id}")

    device = torch.device(args.device)

    """ train_labels = [y for _, y in train_loader.dataset.dataset.samples]  # 原始 dataset 的所有标签
    train_indices = train_loader.dataset.indices                        # 当前 client 的样本索引
    client_labels = [train_labels[i] for i in train_indices]

    label_dist = collections.Counter(client_labels)
    logger.info(f"[Client {client_id}] Data allocation: "
                f"train_size={train_size}, test_size={len(test_loader.dataset)}")
    logger.info(f"[Client {client_id}] Label distribution: {dict(label_dist)}") """

    client_pre_eval_acc = []
    client_pre_eval_loss = []
    client_post_eval_acc = []
    client_post_eval_loss = []

    # 主循环
    uploaded_round = set()
    last_round = -1
    train_size = len(train_loader)
    
    model = None    

    while True:
        task = stub.GetTask(fed_pb2.GetTaskRequest(client_id=client_id))

        if start_time is None and task.round < cfg.total_rounds:
            start_time = time.time()
            logger.info("Training timer started.")

        if task.round == -1:
            break
        if task.round >= cfg.total_rounds:
            logger.info(f"[Client {client_id}] All rounds finished.")
            if start_time is not None and end_time is None:
                end_time = time.time()
                elapsed = end_time - start_time
                logger.info(f"[Client {client_id}] Total training time: {elapsed:.2f}s "
                            f"({elapsed/60:.2f} min)")
                logger.info(f"[Client {client_id}] total acc : {client_pre_eval_acc}")
                logger.info(f"[Client {client_id}] total loss : {client_pre_eval_loss}")
            break

        if task.round != last_round:    
            logger.info(f"[Client {client_id}] Enter round {task.round}")
            last_round = task.round

        if not task.participate:
            time.sleep(1.0)
            continue

        # 拉取并加载全局模型
        # model = create_model(task.config.model_name, num_classes=args.num_classes).to(device)
        if task.round == 0:
            logger.info("Get signal, training start!")
        if not model:
            model = FedEXTModel(client_index, task.config.model_name, args.feature_dim, args.num_classes, args.encoder_ratio).to(device)
        state_dict = bytes_to_state_dict(task.global_model)
        model.load_state_dict(state_dict)

        pre_test_loss, pre_test_acc = evaluate(model, test_loader, device=device)
        logger.info(f"[Client {client_id}][Round {task.round}] (pre-trained) test_acc={pre_test_acc:.4f}")
        client_pre_eval_acc.append(pre_test_acc)
        client_pre_eval_loss.append(pre_test_loss)


        # 本地训练
        optimizer = torch.optim.SGD(model.parameters(), lr=task.config.lr, momentum=task.config.momentum, weight_decay=5e-4)
        train_loss, train_acc = train_local(model, train_loader, epochs=task.config.local_epochs, optimizer=optimizer, device=device)
        test_loss, test_acc = evaluate(model, test_loader, device=device)
        logger.info(f"[Client {client_id}][Round {task.round}] (post-trained) train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

        client_post_eval_acc.append(test_acc)
        client_post_eval_loss.append(test_loss)

        # 回传完整模型（简单实现）
        local_bytes = state_dict_to_bytes(model.state_dict())
        reply = stub.UploadUpdate(
            fed_pb2.UploadRequest(
                client_id=client_id,
                group_id=group_index,
                round=task.round,
                local_model=local_bytes,
                num_samples=train_size,
                test_acc=pre_test_acc,
            )
        )

        if reply.accepted:
            # 继续轮询下一轮
            time.sleep(0.2)
        else:
            # 如果被拒绝（比如轮次错位），稍后重试
            time.sleep(1.0)

    print(f"Client pre acc : {client_pre_eval_acc}")
    print(f"Client post acc : {client_post_eval_acc}")
if __name__ == "__main__":
    main()

