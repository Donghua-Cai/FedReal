# client/client_kd_main.py
import argparse
import collections
import logging
import time

import grpc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from proto import fed_pb2, fed_pb2_grpc
from common.utils import setup_logger
from common.model.create_model import create_model
from common.dataset import (
    build_imagefolder_loaders_for_client,
    train_tf_cifar10,
    test_tf_cifar10,
)
from common.losses import KLDivergenceWithTemperature

# 复用你已有的本地训练/评测函数
from client.trainer import train_local, evaluate


def infer_public_logits(model, public_loader, device="cpu"):
    """对公共无标签集推理 logits（保持与服务器一致顺序：public_loader 必须 shuffle=False）"""
    model.eval()
    outs = []
    with torch.no_grad():
        for x in tqdm(public_loader, desc="Infer public logits", leave=False):
            x = x.to(device)
            out = model(x)
            outs.append(out.cpu())
    return torch.cat(outs, dim=0).contiguous()  # [N, C]


def train_kd_on_public(
    model,
    public_loader,
    teacher_logits_full,
    pseudo_labels_full,
    epochs,
    lr,
    momentum,
    T,
    alpha,
    device="cpu",
):
    """用 server 下发的 teacher logits + 伪标签，在公共集上做 KD 训练"""
    model.train()
    opt = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4
    )
    kd_loss = KLDivergenceWithTemperature(T)

    for ep in range(epochs):
        total = 0
        loss_sum = 0.0
        correct = 0
        offset = 0
        for x in tqdm(public_loader, desc=f"Client KD {ep+1}/{epochs}", leave=False):
            x = x.to(device)
            bsz = x.size(0)
            t_logits = teacher_logits_full[offset : offset + bsz].to(device)
            y_pseudo = pseudo_labels_full[offset : offset + bsz].to(device)
            offset += bsz

            opt.zero_grad()
            s_logits = model(x)
            ce = F.cross_entropy(s_logits, y_pseudo)
            kl = kd_loss(s_logits, t_logits)
            loss = ce + alpha * kl
            loss.backward()
            opt.step()

            loss_sum += loss.item() * bsz
            pred = s_logits.argmax(dim=1)
            correct += (pred == y_pseudo).sum().item()
            total += bsz
        # 需要更详细日志可在此 logger.info 每个 epoch 的 KD 指标


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:50052")
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument(
        "--partition_method", type=str, default="dirichlet", choices=["iid", "dirichlet"]
    )
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = setup_logger(f"Client-{args.client_name}", level=logging.INFO)

    # gRPC 通道
    max_len = 128 * 1024 * 1024
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )
    stub = fed_pb2_grpc.FederatedServiceStub(channel)

    # 注册
    reg = stub.RegisterClient(fed_pb2.RegisterRequest(client_name=args.client_name))
    client_id = reg.client_id
    client_index = reg.client_index
    cfg = reg.config

    logger.info(f"[{client_id}] index={client_index}; device={args.device}")

    # 数据加载（确定性划分；public_loader 必须 shuffle=False）
    train_loader, test_loader, public_loader, train_size, num_classes = (
        build_imagefolder_loaders_for_client(
            data_root=args.data_root,
            dataset_name=args.dataset_name,
            client_index=client_index,
            num_clients=args.num_clients,
            partition_method=args.partition_method,
            dirichlet_alpha=args.dirichlet_alpha,
            batch_size=args.batch_size,
            seed=args.seed,
            train_transform=train_tf_cifar10,
            test_transform=test_tf_cifar10,
            public_ratio=0.1,
            server_test_ratio=0.1,
            return_public_loader=True,
            return_index_in_public=False,
        )
    )

    # 打印本地数据分布
    train_labels_all = [y for _, y in train_loader.dataset.dataset.samples]
    my_train_idx = train_loader.dataset.indices
    my_labels = [train_labels_all[i] for i in my_train_idx]
    lb_dist = collections.Counter(my_labels)
    logger.info(
        f"[{client_id}] Data allocation: train_size={len(my_train_idx)}, "
        f"test_size={len(test_loader.dataset)}"
    )
    logger.info(f"[{client_id}] Label distribution: {dict(lb_dist)}")

    device = torch.device(args.device)
    start_time = None
    end_time = None

    # 本轮状态缓存
    teacher_cache = None  # torch.Tensor [N,C]
    finished_local = False
    finished_kd = False
    last_round = -1

    while True:
        task = stub.GetTask(fed_pb2.GetTaskRequest(client_id=client_id))

        # 强同步：未注册满/服务器未放行时，处于 WAITING 相位，只轮询不训练
        if task.phase == fed_pb2.PHASE_WAITING:
            time.sleep(0.3)
            continue

        # 启动计时（从第一轮 Local Supervised 开始）
        if start_time is None and task.phase == fed_pb2.PHASE_LOCAL_SUP:
            start_time = time.time()
            logger.info("Training timer started.")

        # 训练完成
        if task.phase == fed_pb2.PHASE_DONE or task.round >= cfg.total_rounds:
            if start_time is not None and end_time is None:
                end_time = time.time()
                elapsed = end_time - start_time
                logger.info(
                    f"[{client_id}] Total training time: {elapsed:.2f}s ({elapsed/60:.2f} min)"
                )
            break

        # 新一轮，清理状态
        if task.round != last_round:
            teacher_cache = None
            finished_local = False
            finished_kd = False
            last_round = task.round
            logger.info(f"[{client_id}] Enter round {task.round}")

        # ========== PHASE A：本地监督训练 + 本地测试 + 上传公共集 logits ==========
        if task.phase == fed_pb2.PHASE_LOCAL_SUP and not finished_local:
            # 初始化/加载模型（client 侧模型，例如 resnet18）
            model = create_model(cfg.model_name, num_classes=cfg.num_classes).to(device)

            # 本地监督训练（复用 trainer.py）
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=5e-4,
            )
            tr_loss, tr_acc = train_local(
                model,
                train_loader,
                epochs=cfg.local_epochs,
                optimizer=optimizer,
                device=device,
            )

            # 本地测试（复用 trainer.py）
            te_loss, te_acc = evaluate(model, test_loader, device=device)
            logger.info(
                f"[{client_id}][Round {task.round}] "
                f"local_train_loss={tr_loss:.4f}, local_train_acc={tr_acc:.4f}, "
                f"local_test_loss={te_loss:.4f}, local_test_acc={te_acc:.4f}"
            )

            # 公共集推理，上传 logits
            logits = infer_public_logits(model, public_loader, device=device)  # [N, C]
            logits_bytes = logits.numpy().astype("float32").tobytes()
            N, C = logits.shape
            payload = fed_pb2.PublicLogitsPayload(
                client_id=client_id,
                round=task.round,
                logits=logits_bytes,
                num_classes=C,
                total_examples=N,
                local_train_samples=len(my_train_idx),
            )
            logger.info(
                f"[{client_id}] [Round {task.round}] Upload logits bytes={len(logits_bytes)} "
                f"({len(logits_bytes)/1024/1024:.2f} MB)"
            )
            _ = stub.UploadPublicLogits(iter([payload]))
            finished_local = True
            time.sleep(0.2)
            continue

        # ========== PHASE B：接收 teacher logits -> 公共集 KD 训练 -> 本地测试 -> ACK ==========
        if task.phase == fed_pb2.PHASE_CLIENT_KD and not finished_kd:
            # 收到 teacher logits（只需一次）
            if task.teacher_logits:
                N = cfg.public_examples
                C = cfg.num_classes
                arr = np.frombuffer(task.teacher_logits, dtype=np.float32).copy()  # .copy() 以避免只读警告
                if arr.size != N * C:
                    logger.error(
                        f"[{client_id}] Bad teacher logits size: {arr.size}, expect {N*C}"
                    )
                    time.sleep(0.5)
                    continue
                teacher_cache = torch.from_numpy(arr).view(N, C).contiguous()

            if teacher_cache is None:
                # 还没拿到 teacher logits，继续轮询
                time.sleep(0.2)
                continue

            # 构造伪标签
            pseudo = teacher_cache.argmax(dim=1).to(torch.long)

            # 使用与监督阶段同结构的新模型在公共集做 KD（或复用也可，这里用新实例更干净）
            model = create_model(cfg.model_name, num_classes=cfg.num_classes).to(device)
            train_kd_on_public(
                model,
                public_loader,
                teacher_cache,
                pseudo,
                epochs=cfg.client_kd_epochs,
                lr=cfg.lr,
                momentum=cfg.momentum,
                T=cfg.kd_temperature,
                alpha=cfg.kd_alpha,
                device=device,
            )

            # 可选：KD 后在本地测试集上测一下
            kd_te_loss, kd_te_acc = evaluate(model, test_loader, device=device)
            logger.info(
                f"[{client_id}][Round {task.round}] "
                f"post_KD_test_loss={kd_te_loss:.4f}, post_KD_test_acc={kd_te_acc:.4f}"
            )

            # 发送 ACK（不上传模型，仅确认完成）
            ack = fed_pb2.UploadRequest(
                client_id=client_id,
                round=task.round,
                local_model=b"",  # 本算法 ACK 不上传权重
                num_samples=0,
            )
            _ = stub.UploadUpdate(ack)
            logger.info(f"[{client_id}] [Round {task.round}] KD done, sent ACK.")
            finished_kd = True
            time.sleep(0.2)
            continue

        # 兜底：轻微休眠，避免空转
        time.sleep(0.2)


if __name__ == "__main__":
    main()