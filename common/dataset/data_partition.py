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
    targets: Sequence[int],
    num_users: int,
    num_shards_per_user: int,
    num_classes_per_user: int,
    sample_num_per_shard: int,
    num_classes: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    在给定的 targets（序列；通常是“池”里的标签序列）上进行非IID切片：
      - 将样本按标签排序；
      - 每个类按 sample_num_per_shard 切成若干 shard；
      - 每个用户领取 num_shards_per_user 个 shard，
        涉及的类数受限为 num_classes_per_user；
      - 选类方式：用户 i 从 (i * num_classes_per_user) % num_classes 开始取连续的 num_classes_per_user 个类；
      - 每个用户的样本总数一致：num_shards_per_user * sample_num_per_shard；
    返回值为：user_id -> 该“池”内索引（注意：这些索引是相对传入 targets 的位置）。
    """
    assert num_classes_per_user >= 1
    assert num_shards_per_user >= num_classes_per_user, "num_shards_per_user 必须 >= num_classes_per_user"
    assert sample_num_per_shard > 0

    # —— 在“池”空间中按标签排序 —— #
    idxs = np.arange(len(targets), dtype=np.int64)
    labels = np.array([int(t) for t in targets], dtype=np.int64)
    order = np.argsort(labels)
    idxs = idxs[order]
    labels = labels[order]

    # —— 统计各类起始位置和样本数 —— #
    class_counts = [(labels == c).sum() for c in range(num_classes)]
    class_counts = [int(x) for x in class_counts]
    # 如果某个类在池中没有样本，也允许（意味着该类对所有用户都不可分配）
    class_shards_num = np.array([cnt // sample_num_per_shard for cnt in class_counts], dtype=np.int64)

    # 供给/需求检查
    total_supply = int(class_shards_num.sum())
    total_demand = num_users * num_shards_per_user
    if total_supply < total_demand:
        raise ValueError(
            f"[shards] 供给不足：可切 shard={total_supply}，需求={total_demand}。"
            f"请减小 sample_num_per_shard 或 num_users/num_shards_per_user。"
        )

    # 各类的“排序后起点”
    labels_start = []
    for c in range(num_classes):
        pos = np.where(labels == c)[0]
        labels_start.append(int(pos[0]) if pos.size > 0 else -1)

    # 每类可分配的 shard 编号池
    available: List[List[int]] = [
        list(range(int(class_shards_num[c]))) for c in range(num_classes)
    ]

    # 把 num_shards_per_user 尽量均匀分到 num_classes_per_user 个类
    base = num_shards_per_user // num_classes_per_user
    rem = num_shards_per_user % num_classes_per_user
    take_list = [base + (1 if i < rem else 0) for i in range(num_classes_per_user)]
    assert sum(take_list) == num_shards_per_user

    rng = np.random.default_rng(seed)
    mapping: Dict[int, List[int]] = {u: [] for u in range(num_users)}

    for uid in range(num_users):
        start_c = (uid * num_classes_per_user) % num_classes
        for s in range(num_classes_per_user):
            c = (start_c + s) % num_classes
            take = take_list[s]
            if class_shards_num[c] == 0:
                # 该类在池中没有样本，无法分配；直接跳过（也可选择向后回退找下一个有供给的类，
                # 这里采用“强约束”策略，抛错更直观）
                raise RuntimeError(
                    f"[shards] 类 {c} 在该池中无可用 shard（count={class_counts[c]}），"
                    f"uid={uid} 需要 {take} 个。请调整参数或上层 public_ratio。"
                )
            if len(available[c]) < take:
                raise RuntimeError(
                    f"[shards] 类 {c} shard 不足：剩余 {len(available[c])} < 需求 {take}；"
                    f"uid={uid}，请调整参数。"
                )
            chosen = rng.choice(available[c], size=take, replace=False).tolist()
            remain = list(set(available[c]) - set(chosen))
            remain.sort()
            available[c] = remain

            c_start = labels_start[c]
            assert c_start >= 0, f"class {c} start not found but class_shards_num>0"
            for shard_id in chosen:
                a = c_start + shard_id * sample_num_per_shard
                b = c_start + (shard_id + 1) * sample_num_per_shard
                mapping[uid].extend(idxs[a:b].tolist())

        rng.shuffle(mapping[uid])

        expect = num_shards_per_user * sample_num_per_shard
        if len(mapping[uid]) != expect:
            raise RuntimeError(f"[shards] uid={uid} 数量异常：got={len(mapping[uid])}, expect={expect}")

    return mapping

def split_train_test(indices: List[int], test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_ratio))
    return idx[n_test:], idx[:n_test]