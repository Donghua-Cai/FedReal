# client/client_main.py
import argparse
import time
import grpc
import torch
import logging
import collections
import torch.nn.functional as F
from tqdm import tqdm

from proto import fed_pb2, fed_pb2_grpc
from common.dataset import build_imagefolder_loaders_for_client
from common.dataset.data_transform import get_transform
from common.model.create_model import create_model
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.utils import setup_logger, set_seed
from client.trainer import train_local, evaluate


def infer_public_logits(model, public_loader, device="cpu"):
    """在公共无标签集上前向推理，收集 [N, C] logits。"""
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in tqdm(public_loader, desc="Infer public logits", leave=False):
            # 兼容 public_loader 可能返回 (img, idx) 或 img
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            out = model(x)
            outs.append(out.cpu())
    return torch.cat(outs, dim=0).contiguous()  # [N, C]


def kd_train_on_public(model, public_loader, teacher_logits_full, epochs, T, alpha, lr, momentum, device="cpu"):
    """
    在公共无标签集上做 KD 训练：
    L = (1 - alpha) * CE(y_pseudo, s_logit) + alpha * KL(softmax(t/T), softmax(s/T)) * T^2
    """
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    N, C = teacher_logits_full.shape

    for ep in range(epochs):
        total, loss_sum, correct = 0, 0.0, 0
        offset = 0
        for batch in tqdm(public_loader, desc=f"Client KD {ep+1}/{epochs}", leave=False):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            bsz = x.size(0)

            t_logits = teacher_logits_full[offset: offset + bsz].to(device)
            y_pseudo = t_logits.argmax(dim=1)
            offset += bsz

            opt.zero_grad()
            s_logits = model(x)
            ce = F.cross_entropy(s_logits, y_pseudo)
            kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            loss = (1.0 - alpha) * ce + alpha * kl
            loss.backward()
            opt.step()

            loss_sum += loss.item() * bsz
            pred = s_logits.argmax(dim=1)
            correct += (pred == y_pseudo).sum().item()
            total += bsz
        # 如需打印每个 epoch 的 KD 指标，可解除注释：
        # print(f"[ClientKD] epoch {ep+1}/{epochs} loss={loss_sum/total:.4f} acc={correct/total:.4f}")


def main():
    start_time = None
    end_time = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:50051")
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_name", type=str, default="cifar10")

    # 拆分/数据
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--partition_method", type=str, default="iid", choices=["iid", "dirichlet", "shards"])
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--client_test_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # shards 参数（仅在 partition_method=shards 时使用）
    parser.add_argument("--sample_num_per_shard", type=int, default=60)
    parser.add_argument("--num_shards_per_user", type=int, default=8)
    parser.add_argument("--num_classes_per_user", type=int, default=3)
    parser.add_argument("--sample_num_per_shard_test", type=int, default=12)

    # 分组个数（用于按标签相似性分组 & group_id 下发/上传）
    parser.add_argument("--group_num", type=int, default=5)

    args = parser.parse_args()

    set_seed(args.seed)
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

    # 构建数据（保持与服务端相同的 seed / transform / 划分比例）
    train_loader, test_loader, public_loader, train_size, num_classes, group_id = build_imagefolder_loaders_for_client(
        data_root=args.data_root,
        dataset_name=args.dataset_name,
        client_index=client_index,
        num_clients=args.num_clients,
        partition_method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        train_transform=get_transform(args.dataset_name, "train"),
        test_transform=get_transform(args.dataset_name, "test"),
        public_ratio=0.1,
        server_test_ratio=0.1,
        return_public_loader=True,
        return_index_in_public=False,
        sample_num_per_shard=args.sample_num_per_shard,
        num_shards_per_user=args.num_shards_per_user,
        num_classes_per_user=args.num_classes_per_user,
        sample_num_per_shard_test=args.sample_num_per_shard_test,
        group_num=args.group_num,
    )
    logger.info(f"[{client_id}] group_id={group_id}")

    # 打印本地标签分布
    try:
        # train_loader.dataset 可能是 Subset -> .dataset 才是 ImageFolder
        base_ds = train_loader.dataset.dataset
        all_labels = [y for _, y in base_ds.samples]
        my_idx = train_loader.dataset.indices
        my_labels = [all_labels[i] for i in my_idx]
        label_dist = collections.Counter(my_labels)

        base_ds_test = test_loader.dataset.dataset
        all_labels_test = [y for _, y in base_ds_test.samples]
        my_idx_test = test_loader.dataset.indices
        my_labels_test = [all_labels_test[i] for i in my_idx_test]
        label_dist_test = collections.Counter(my_labels_test)

        logger.info(f"[{client_id}] Data allocation: train_size={len(my_idx)}, test_size={len(test_loader.dataset)}")
        logger.info(f"[{client_id}] Train label distribution: {dict(label_dist)}")
        logger.info(f"[{client_id}] Test label distribution: {dict(label_dist_test)}")
    except Exception:
        # 兜底（有些自定义数据集可能没有 .samples）
        logger.info(f"[{client_id}] Data allocation: train_size={train_size}, test_size={len(test_loader.dataset)}")

    device = torch.device(args.device)

    # 每轮测试记录
    client_eval_acc, client_eval_loss = [], []

    # 一轮一次闸门，防止在同一 phase 内重复上传
    done_local_rounds = set()
    sent_group_logits_rounds = set()
    acked_kd_rounds = set()

    last_round = -1
    while True:
        task = stub.GetTask(fed_pb2.GetTaskRequest(client_id=client_id))

        # 训练计时
        if start_time is None and task.phase == fed_pb2.PHASE_LOCAL_TRAIN and task.round < cfg.total_rounds:
            start_time = time.time()
            logger.info("Training timer started.")

        # 结束
        if task.phase == fed_pb2.PHASE_DONE or task.round >= cfg.total_rounds:
            logger.info(f"[{client_id}] All rounds finished.")
            if start_time is not None and end_time is None:
                end_time = time.time()
                elapsed = end_time - start_time
                logger.info(f"[{client_id}] Total training time: {elapsed:.2f}s ({elapsed/60:.2f} min)")
                logger.info(f"[{client_id}] total acc : {client_eval_acc}")
                logger.info(f"[{client_id}] total loss: {client_eval_loss}")
            break

        # 新一轮提示
        if task.round != last_round:
            logger.info(f"[{client_id}] Enter round {task.round}")
            last_round = task.round

        # 没活/等待
        if task.phase == fed_pb2.PHASE_WAITING or not task.participate:
            time.sleep(0.3)
            continue

        # 读取 KD 超参（若 proto 未含这些字段，服务端也会在 config 里带默认）
        kd_T = getattr(task.config, "kd_temperature", 1.0)
        kd_alpha = getattr(task.config, "kd_alpha", 0.5)
        client_kd_epochs = getattr(task.config, "client_kd_epochs", 3)

        # ========== 阶段 A：本地监督训练 + 上传（带 group_id） ==========
        if task.phase == fed_pb2.PHASE_LOCAL_TRAIN:
            # 一轮只训练/上传一次
            if task.round in done_local_rounds:
                time.sleep(0.2)
                continue

            model = create_model(task.config.model_name, num_classes=task.config.num_classes).to(device)
            # 下发字段：group_model（非 global_model）
            if task.group_model:
                try:
                    state_dict = bytes_to_state_dict(task.group_model)
                    model.load_state_dict(state_dict)
                except Exception:
                    pass  # 初轮可能为空/结构不匹配

            optimizer = torch.optim.SGD(
                model.parameters(), lr=task.config.lr,
                momentum=task.config.momentum, weight_decay=5e-4
            )
            train_loss, train_acc = train_local(
                model, train_loader, epochs=task.config.local_epochs,
                optimizer=optimizer, device=device
            )
            test_loss, test_acc = evaluate(model, test_loader, device=device)
            logger.info(f"Round {task.round} - [{client_id}] group_id={group_id} eval acc : {test_acc}, loss : {test_loss}")
            client_eval_acc.append(test_acc)
            client_eval_loss.append(test_loss)

            local_bytes = state_dict_to_bytes(model.state_dict())
            _ = stub.UploadUpdate(fed_pb2.UploadRequest(
                client_id=client_id,
                round=task.round,
                local_model=local_bytes,
                num_samples=len(train_loader.dataset),
                group_id=group_id,
                train_loss=train_loss, train_acc=train_acc,
                test_loss=test_loss,   test_acc=test_acc,
            ))
            done_local_rounds.add(task.round)
            time.sleep(0.2)
            continue

        # ========== 阶段 B：组聚合（仅接收，不需要动作） ==========
        if task.phase == fed_pb2.PHASE_GROUP_AGG:
            time.sleep(0.2)
            continue

        # ========== 阶段 C：组代表上传 logits ==========
        if task.phase == fed_pb2.PHASE_GROUP_LOGITS:
            # 一轮只上传一次
            if task.round in sent_group_logits_rounds:
                time.sleep(0.2)
                continue

            model = create_model(task.config.model_name, num_classes=task.config.num_classes).to(device)
            # 为了提升 logits 质量，尽可能加载“组聚合后的小模型”
            if task.group_model:
                try:
                    model.load_state_dict(bytes_to_state_dict(task.group_model))
                except Exception:
                    pass

            logits = infer_public_logits(model, public_loader, device=device)  # [N, C]
            import numpy as np  # 局部引入，避免顶部无用依赖
            logits_bytes = logits.numpy().astype("float32").tobytes()
            total_examples = getattr(task.config, "public_examples", len(public_loader.dataset))

            # ✅ 使用一元 RPC：UploadGroupLogits
            payload = fed_pb2.GroupLogits(
                client_id=client_id,
                group_id=group_id,
                round=task.round,
                logits=logits_bytes,
                num_classes=task.config.num_classes,
                total_examples=total_examples,
            )
            _ = stub.UploadGroupLogits(payload)

            sent_group_logits_rounds.add(task.round)
            time.sleep(0.2)
            continue

        # ========== 阶段 D：接收 server_logits -> 本地 KD -> 发送 ACK（空模型） ==========
        if task.phase == fed_pb2.PHASE_CLIENT_KD:
            if task.round in acked_kd_rounds:
                time.sleep(0.2)
                continue
            if not task.server_logits:
                time.sleep(0.2)
                continue

            import numpy as np
            N = getattr(task.config, "public_examples", len(public_loader.dataset))
            C = task.config.num_classes
            arr = np.frombuffer(task.server_logits, dtype=np.float32).copy()
            if arr.size != N * C:
                # 尺寸不对，等下一次轮询
                time.sleep(0.3)
                continue
            teacher = torch.from_numpy(arr).view(N, C).contiguous()
            teacher.requires_grad_(False)

            model = create_model(task.config.model_name, num_classes=task.config.num_classes).to(device)
            kd_train_on_public(
                model, public_loader, teacher,
                epochs=client_kd_epochs, T=kd_T, alpha=kd_alpha,
                lr=task.config.lr, momentum=task.config.momentum, device=device
            )

            # 发送 ACK（空模型即表示完成）
            _ = stub.UploadUpdate(fed_pb2.UploadRequest(
                client_id=client_id,
                round=task.round,
                local_model=b"",
                num_samples=0,
                group_id=group_id,
            ))
            acked_kd_rounds.add(task.round)
            time.sleep(0.2)
            continue

        # 兜底
        time.sleep(0.2)


if __name__ == "__main__":
    main()