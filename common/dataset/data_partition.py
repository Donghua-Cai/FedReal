# common/dataset/data_partition.py
from typing import Dict, List, Tuple, Sequence
import random
import numpy as np

__all__ = ["iid_partition", "dirichlet_partition", "split_train_test", "noniid_shards_partition"]

def _class_indices(targets: Sequence[int], num_classes: int) -> List[List[int]]:
    per_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(targets):
        per_class[int(y)].append(idx)
    return per_class

def iid_partition(targets: Sequence[int], num_clients: int, seed: int) -> Dict[int, List[int]]:
    num_classes = len(set(int(t) for t in targets))
    per_class = _class_indices(targets, num_classes)
    rng = random.Random(seed)
    for cls in range(num_classes):
        rng.shuffle(per_class[cls])

    client_indices = {cid: [] for cid in range(num_clients)}
    for cls in range(num_classes):
        parts = np.array_split(per_class[cls], num_clients)
        for cid, arr in enumerate(parts):
            client_indices[cid].extend(arr.tolist())

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices

def dirichlet_partition(targets: Sequence[int], num_clients: int, alpha: float, seed: int) -> Dict[int, List[int]]:
    num_classes = len(set(int(t) for t in targets))
    per_class = _class_indices(targets, num_classes)
    rng = np.random.default_rng(seed)

    client_indices = {cid: [] for cid in range(num_clients)}
    for cls in range(num_classes):
        cls_idx = per_class[cls][:]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        p = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        splits = (np.cumsum(p) * n).astype(int).clip(0, n)
        prev = 0
        for cid, s in enumerate(splits):
            client_indices[cid].extend(cls_idx[prev:s])
            prev = s
        if prev < n:
            client_indices[num_clients - 1].extend(cls_idx[prev:])

    for cid in range(num_clients):
        random.Random(seed + cid).shuffle(client_indices[cid])
    return client_indices

def noniid_shards_partition(
    *,
    train_targets: Sequence[int],
    test_targets: Sequence[int],
    num_users: int,
    num_classes: int,
    sample_num_per_shard: int = 60,
    num_shards_per_user: int = 8,
    num_classes_per_user: int = 3,
    sample_num_per_shard_test: int = 12,
    seed: int = 42,
    group_num: int = 5,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]:
    """
    参考你给的 shards 逻辑：
    - 每类被切成若干 train/test shard；每个 client 取 num_shards_per_user 个 shard，
      其中只覆盖 num_classes_per_user 个类别。
    - 额外：按各 client 的标签直方图（训练集）相似性做 K-Means 聚成 group_num 组，返回 group_id。

    返回:
      dict_users_train: cid -> train indices
      dict_users_test:  cid -> test indices
      client_group_id:  cid -> group_id in [0, group_num-1]
    """
    rng = np.random.default_rng(seed)

    # —— 构造“每类样本的有序索引”（按标签排序后切片方便按 shard 取）——
    idxs_train = np.arange(len(train_targets))
    idxs_test  = np.arange(len(test_targets))

    labels_train = np.asarray(train_targets, dtype=np.int64)
    labels_test  = np.asarray(test_targets, dtype=np.int64)

    # 按标签排序
    order_tr = np.argsort(labels_train)
    order_te = np.argsort(labels_test)
    idxs_train = idxs_train[order_tr]
    labels_train = labels_train[order_tr]
    idxs_test = idxs_test[order_te]
    labels_test = labels_test[order_te]

    # 每类起始位置 + 每类样本数
    labels_start_idx_train = [int(np.where(labels_train == c)[0][0]) for c in range(num_classes)]
    labels_start_idx_test  = [int(np.where(labels_test  == c)[0][0]) for c in range(num_classes)]
    class_idx_num_train = [int((labels_train == c).sum()) for c in range(num_classes)]
    class_idx_num_test  = [int((labels_test  == c).sum()) for c in range(num_classes)]

    # 每类能切出多少个 shard（整除向下）
    class_shards_num_train = np.array(class_idx_num_train) // int(sample_num_per_shard)
    class_shards_num_test  = np.array(class_idx_num_test)  // int(sample_num_per_shard_test)

    # 可用 shard 编号池（每类）
    available_class_shard_train = [list(range(int(class_shards_num_train[c]))) for c in range(num_classes)]
    available_class_shard_test  = [list(range(int(class_shards_num_test[c])))  for c in range(num_classes)]

    # 将 num_shards_per_user 在 num_classes_per_user 个类中“平均”分配
    a = list(range(int(num_shards_per_user)))
    shard_idx_per_class = [
        a[x * num_shards_per_user // num_classes_per_user : (x + 1) * num_shards_per_user // num_classes_per_user]
        for x in range(int(num_classes_per_user))
    ]
    selected_shards_num_per_class = [len(s) for s in shard_idx_per_class]
    assert len(selected_shards_num_per_class) == num_classes_per_user
    assert np.sum(selected_shards_num_per_class) == num_shards_per_user

    dict_users_train: Dict[int, List[int]] = {i: [] for i in range(num_users)}
    dict_users_test:  Dict[int, List[int]] = {i: [] for i in range(num_users)}

    # —— 分配给每个 client：连续类轮换（和你给的逻辑一致）——
    for i in range(num_users):
        selected_class = (num_classes_per_user * i) % num_classes
        for s in range(num_classes_per_user):
            c = int((selected_class + s) % num_classes)
            need_train = selected_shards_num_per_class[s]
            need_test  = selected_shards_num_per_class[s]

            # 采样本类中未使用的 shard id
            if len(available_class_shard_train[c]) < need_train:
                raise RuntimeError(f"Class {c} has not enough train shards left.")
            if len(available_class_shard_test[c]) < need_test:
                raise RuntimeError(f"Class {c} has not enough test shards left.")

            sel_tr = rng.choice(available_class_shard_train[c], need_train, replace=False).tolist()
            sel_te = rng.choice(available_class_shard_test[c],  need_test,  replace=False).tolist()

            # 标记为占用
            available_class_shard_train[c] = list(set(available_class_shard_train[c]) - set(sel_tr))
            available_class_shard_test[c]  = list(set(available_class_shard_test[c])  - set(sel_te))

            # 采样区间并收集索引
            for shard in sel_tr:
                s0 = labels_start_idx_train[c] + shard * sample_num_per_shard
                s1 = s0 + sample_num_per_shard
                dict_users_train[i].extend(idxs_train[s0:s1].tolist())
            for shard in sel_te:
                s0 = labels_start_idx_test[c] + shard * sample_num_per_shard_test
                s1 = s0 + sample_num_per_shard_test
                dict_users_test[i].extend(idxs_test[s0:s1].tolist())

    # —— 基于训练集标签直方图进行分组（K-Means, k=group_num）——
    # 构造每个 client 的 num_classes 维计数向量
    mats = np.zeros((num_users, num_classes), dtype=np.float32)
    for i in range(num_users):
        lbls = np.asarray([train_targets[j] for j in dict_users_train[i]], dtype=np.int64)
        if lbls.size:
            counts = np.bincount(lbls, minlength=num_classes)
            mats[i] = counts / max(1, counts.sum())  # L1 归一化
        else:
            mats[i] = 0

    k = max(1, int(group_num))
    # 初始化中心（选择前 k 个客户端，或随机）
    if num_users >= k:
        centers = mats[:k].copy()
    else:
        # 不足 k 个客户端时，重复一些
        reps = int(np.ceil(k / max(1, num_users)))
        centers = np.vstack([mats] * reps)[:k]

    # 简单 K-Means 迭代
    for _ in range(10):
        # 分配
        # 使用欧氏距离最小
        d = np.sum((mats[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # [num_users, k]
        assign = np.argmin(d, axis=1)  # [num_users]
        # 更新
        new_centers = np.zeros_like(centers)
        for gid in range(k):
            mask = (assign == gid)
            if np.any(mask):
                new_centers[gid] = mats[mask].mean(axis=0)
            else:
                # 空簇：就拿一个样本硬塞
                ridx = rng.integers(0, num_users)
                new_centers[gid] = mats[ridx]
        # 收敛检查（可选）
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    client_group_id = {i: int(assign[i]) for i in range(num_users)}
    return dict_users_train, dict_users_test, client_group_id

def split_train_test(indices: List[int], test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_ratio))
    return idx[n_test:], idx[:n_test]