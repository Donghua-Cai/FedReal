import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from .data_partition import (
    iid_partition,
    dirichlet_partition,
    split_train_test,
    noniid_shards_partition,  
)
from common.dataset.data_transform import get_transform

__all__ = [
    "make_global_loaders",                 # 服务器：公共无标签 + 服务器测试
    "build_imagefolder_loaders_for_client" # 客户端：本地 train/test + （可选）公共无标签
]

# ---------- 工具 ----------

def _decide_loader_runtime(num_workers: Optional[int], pin_memory: Optional[bool]) -> Tuple[int, bool]:
    has_cuda = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 2 if has_cuda else 0
    if pin_memory is None:
        pin_memory = True if has_cuda else False
    return num_workers, pin_memory

def _pick_eval_dir(root: str, dataset_name: str) -> Optional[str]:
    """优先 val/，否则 test/，都没有则返回 None。"""
    for candidate in ["val", "test"]:
        path = os.path.join(root, dataset_name, candidate)
        if os.path.isdir(path):
            return path
    return None

def _targets_from_imagefolder(img_folder: datasets.ImageFolder) -> List[int]:
    return [y for _, y in img_folder.samples]

class UnlabeledSubset(Dataset):
    """只返回图像（无标签），可选返回样本在 base 内的索引。"""
    def __init__(self, base: Dataset, indices: Sequence[int], return_index: bool = False):
        self.base = base
        self.indices = list(indices)
        self.return_index = return_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, _ = self.base[idx]  # 丢弃标签
        if self.return_index:
            return img, idx
        return img

# ---------- 计算“公共/服务器测试/剩余池” ----------

def _compute_global_pools(
    train_set: datasets.ImageFolder,
    eval_set: Optional[datasets.ImageFolder],
    public_ratio: float,
    server_test_ratio: float,
    seed: int,
) -> Tuple[List[int], Optional[List[int]], List[int], Optional[List[int]]]:
    """
    返回：
      public_indices         —— 从 train/ 抽取的无标签公共集
      server_test_indices    —— 从 val|test/ 抽取的服务器测试集（可能为 None）
      client_train_pool      —— train/ 剩余给 client 划分的池
      client_test_pool       —— val|test/ 剩余给 client 划分的池（可能为 None）
    """
    rng = random.Random(seed)

    # public from train
    all_train_idx = list(range(len(train_set)))
    rng.shuffle(all_train_idx)
    n_public = max(1, int(len(all_train_idx) * public_ratio))
    public_indices = all_train_idx[:n_public]
    client_train_pool = all_train_idx[n_public:]

    # server test from val/test (optional)
    server_test_indices, client_test_pool = None, None
    if eval_set is not None and len(eval_set) > 0:
        all_eval_idx = list(range(len(eval_set)))
        rng.shuffle(all_eval_idx)
        n_server_test = max(1, int(len(all_eval_idx) * server_test_ratio))
        server_test_indices = all_eval_idx[:n_server_test]
        client_test_pool = all_eval_idx[n_server_test:]

    return public_indices, server_test_indices, client_train_pool, client_test_pool

# ---------- 服务器端：公共无标签 + 服务器测试 ----------

def make_global_loaders(
    data_root: str,
    dataset_name: str,
    batch_size: int = 64,
    seed: int = 42,
    public_ratio: float = 0.1,
    server_test_ratio: float = 0.1,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    train_transform=None,
    test_transform=None,
    return_index_in_public: bool = False,
):
    """
    返回 (public_unlabeled_loader, server_test_loader)
    - public_unlabeled_loader: 从 train/ 抽取的一份无标签集（所有端可共享）
    - server_test_loader: 从 val|test/ 抽取的 10% 子集（仅服务器评测用），可能为 None
    """
    train_dir = os.path.join(data_root, dataset_name, "train")
    eval_dir = _pick_eval_dir(data_root, dataset_name)  # val or test

    train_set = datasets.ImageFolder(train_dir, transform=train_transform or get_transform(dataset_name, "train"))
    eval_set = datasets.ImageFolder(eval_dir, transform=test_transform or get_transform(dataset_name, "test")) if eval_dir else None

    public_idx, server_test_idx, _, _ = _compute_global_pools(
        train_set, eval_set, public_ratio, server_test_ratio, seed
    )

    # 无标签公共集
    public_subset = UnlabeledSubset(train_set, public_idx, return_index=return_index_in_public)

    nw, pm = _decide_loader_runtime(num_workers, pin_memory)
    public_loader = DataLoader(public_subset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    server_test_loader = None
    if eval_set is not None and server_test_idx is not None:
        server_test_subset = Subset(eval_set, server_test_idx)
        server_test_loader = DataLoader(server_test_subset, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    return public_loader, server_test_loader

# ---------- 客户端：在“剩余池”上继续按 IID/Dirichlet/Shards 划分 ----------

def build_imagefolder_loaders_for_client(
    data_root: str,
    dataset_name: str,
    client_index: int,
    num_clients: int,
    partition_method: str = "iid",                 # "iid" | "dirichlet" | "shards"
    dirichlet_alpha: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    # transform
    train_transform=None,
    test_transform=None,
    # 与服务器保持一致的全局划分比例
    public_ratio: float = 0.1,
    server_test_ratio: float = 0.1,
    # 是否也返回公共无标签 Loader
    return_public_loader: bool = True,
    return_index_in_public: bool = False,
    # 仅当 partition_method="shards" 时使用的超参：
    sample_num_per_shard: int = 60,
    num_shards_per_user: int = 8,
    num_classes_per_user: int = 3,
    sample_num_per_shard_test: Optional[int] = None,   # 若为 None，默认=sample_num_per_shard//5 或 1
):
    """
    返回：
      train_loader, test_loader, public_unlabeled_loader(可选), train_size, num_classes
    - 客户端训练/测试来自“剩余池”的再划分（IID/Dirichlet/Shards）
    - 公共无标签是从 train/ 的 public_ratio 抽取（与服务器一致，seed 决定性）
    """
    train_dir = os.path.join(data_root, dataset_name, "train")
    eval_dir = _pick_eval_dir(data_root, dataset_name)

    train_set = datasets.ImageFolder(train_dir, transform=train_transform or get_transform(dataset_name, "train"))
    eval_set = datasets.ImageFolder(eval_dir, transform=test_transform or get_transform(dataset_name, "test")) if eval_dir else None

    # 计算全局池
    public_idx, server_test_idx, client_train_pool, client_test_pool = _compute_global_pools(
        train_set, eval_set, public_ratio, server_test_ratio, seed
    )

    num_classes = len(train_set.classes)

    # 统计剩余池的 targets，用于划分（注意：这里的 targets 是“池内顺序”的标签序列）
    def mask_targets(base_set: datasets.ImageFolder, pool_idx: List[int]) -> List[int]:
        targets_all = _targets_from_imagefolder(base_set)
        return [int(targets_all[i]) for i in pool_idx]

    # 训练池划分到各 client
    train_pool_targets = mask_targets(train_set, client_train_pool)
    if partition_method == "iid":
        mapping_train = iid_partition(train_pool_targets, num_clients, seed)
    elif partition_method == "dirichlet":
        mapping_train = dirichlet_partition(train_pool_targets, num_clients, dirichlet_alpha, seed)
    elif partition_method == "shards":
        mapping_train = noniid_shards_partition(
            targets=train_pool_targets,
            num_users=num_clients,
            num_shards_per_user=num_shards_per_user,
            num_classes_per_user=num_classes_per_user,
            sample_num_per_shard=sample_num_per_shard,
            num_classes=num_classes,
            seed=seed,
        )
    else:
        raise ValueError("partition_method must be 'iid' or 'dirichlet' or 'shards'")

    # 将“池内相对索引”映射回数据集全局索引
    my_train_indices = [client_train_pool[i] for i in mapping_train[client_index]]

    # 测试池划分到各 client（如果有 eval_set）
    my_test_indices: List[int] = []
    if eval_set is not None and client_test_pool is not None and len(client_test_pool) > 0:
        test_pool_targets = mask_targets(eval_set, client_test_pool)
        if partition_method == "iid":
            mapping_test = iid_partition(test_pool_targets, num_clients, seed + 999)
        elif partition_method == "dirichlet":
            mapping_test = dirichlet_partition(test_pool_targets, num_clients, dirichlet_alpha, seed + 999)
        else:  # shards
            sps_t = sample_num_per_shard_test
            if sps_t is None:
                sps_t = max(1, sample_num_per_shard // 5)  # 常见做法：测试 shard 更小
            mapping_test = noniid_shards_partition(
                targets=test_pool_targets,
                num_users=num_clients,
                num_shards_per_user=num_shards_per_user,
                num_classes_per_user=num_classes_per_user,
                sample_num_per_shard=sps_t,
                num_classes=num_classes,
                seed=seed + 999,
            )
        my_test_indices = [client_test_pool[i] for i in mapping_test[client_index]]

    # 组建 DataLoader（固定 shuffle 顺序，不改全局seed）
    nw, pm = _decide_loader_runtime(num_workers, pin_memory)
    train_subset = Subset(train_set, my_train_indices)

    # 用 client_index & seed 组合出稳定的 DataLoader 随机源，固定 shuffle 顺序
    dl_seed = (seed + 100003 * client_index) & 0x7FFFFFFF
    dl_gen = torch.Generator()
    dl_gen.manual_seed(dl_seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=pm,
        generator=dl_gen,  # 关键：固定 DataLoader 的随机打乱
    )

    if len(my_test_indices) > 0:
        test_subset = Subset(eval_set, my_test_indices)
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pm
        )
    else:
        # 如果没有 val/test，退化为在训练池中再切 10% 做本地 test（可复现）
        tr_idx, te_idx = split_train_test(my_train_indices, test_ratio=0.1, seed=seed + client_index + 123)
        train_subset = Subset(train_set, tr_idx)
        test_subset = Subset(train_set, te_idx)

        # 退化分支里也固定训练集 shuffle 顺序
        dl_seed2 = (seed + 100003 * client_index + 1) & 0x7FFFFFFF
        dl_gen2 = torch.Generator().manual_seed(dl_seed2)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=pm,
            generator=dl_gen2
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pm
        )

    public_loader = None
    if return_public_loader:
        public_subset = UnlabeledSubset(train_set, public_idx, return_index=return_index_in_public)
        public_loader = DataLoader(
            public_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pm
        )

    num_classes = len(train_set.classes)
    train_size = len(train_subset)

    return train_loader, test_loader, public_loader, train_size, num_classes