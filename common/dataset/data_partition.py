# common/dataset/data_partition.py
from typing import Dict, List, Tuple, Sequence
import random
import numpy as np

__all__ = ["iid_partition", "dirichlet_partition", "split_train_test"]

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

def split_train_test(indices: List[int], test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_ratio))
    return idx[n_test:], idx[:n_test]