from typing import Dict, List, Tuple
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# —— CIFAR-10 预处理 ——
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def _class_indices(targets: List[int], num_classes: int = 10) -> List[List[int]]:
    per_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(targets):
        per_class[y].append(idx)
    return per_class


def iid_partition(targets: List[int], num_clients: int, seed: int) -> Dict[int, List[int]]:
    # 按类均匀划分，提升稳定性
    num_classes = len(set(targets))
    per_class = _class_indices(targets, num_classes)
    rng = random.Random(seed)
    for cls in range(num_classes):
        rng.shuffle(per_class[cls])

    client_indices = {cid: [] for cid in range(num_clients)}
    for cls in range(num_classes):
        parts = np.array_split(per_class[cls], num_clients)
        for cid, arr in enumerate(parts):
            client_indices[cid].extend(arr.tolist())

    # 每个客户端内部再打乱
    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])
    return client_indices


def dirichlet_partition(targets: List[int], num_clients: int, alpha: float, seed: int) -> Dict[int, List[int]]:
    num_classes = len(set(targets))
    per_class = _class_indices(targets, num_classes)
    rng = np.random.default_rng(seed)

    client_indices = {cid: [] for cid in range(num_clients)}

    for cls in range(num_classes):
        cls_idx = per_class[cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        # 为该类在 num_clients 上采样比例
        proportions = rng.dirichlet(alpha=np.ones(num_clients) * alpha)
        # 按比例切分
        splits = (np.cumsum(proportions) * n).astype(int)
        splits = np.clip(splits, 0, n)
        prev = 0
        for cid, s in enumerate(splits):
            client_indices[cid].extend(cls_idx[prev:s])
            prev = s
        # 剩余给最后一个客户端
        if prev < n:
            client_indices[num_clients - 1].extend(cls_idx[prev:])

    # 打乱
    for cid in range(num_clients):
        random.Random(seed + cid).shuffle(client_indices[cid])

    return client_indices


def split_train_test(indices: List[int], test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(indices)
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_ratio))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return train_idx, test_idx


def build_cifar10_loaders_for_client(
    data_root: str,
    client_index: int,
    num_clients: int,
    partition_method: str = "iid",
    dirichlet_alpha: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 2,
    client_test_ratio: float = 0.1,
):
    # 公共测试集使用官方 test split
    public_test = datasets.CIFAR10(root=data_root, train=False, download=False, transform=TEST_TRANSFORM)

    train_set_full = datasets.CIFAR10(root=data_root, train=True, download=False, transform=TRAIN_TRANSFORM)
    targets = train_set_full.targets  # length 50000

    if partition_method == "iid":
        mapping = iid_partition(targets, num_clients=num_clients, seed=seed)
    elif partition_method == "dirichlet":
        mapping = dirichlet_partition(targets, num_clients=num_clients, alpha=dirichlet_alpha, seed=seed)
    else:
        raise ValueError("partition_method must be 'iid' or 'dirichlet'")

    my_indices = mapping[client_index]
    train_idx, test_idx = split_train_test(my_indices, test_ratio=client_test_ratio, seed=seed + client_index)

    train_subset = Subset(train_set_full, train_idx)
    test_subset = Subset(datasets.CIFAR10(root=data_root, train=True, download=False, transform=TEST_TRANSFORM), test_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    public_test_loader = DataLoader(public_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, public_test_loader, len(train_subset)