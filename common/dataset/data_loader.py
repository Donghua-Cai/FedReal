# common/dataset/data_loader.py
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .data_partition import iid_partition, dirichlet_partition, split_train_test

from typing import Optional, Tuple

__all__ = ["build_imagefolder_loaders_for_client"]

def _decide_loader_runtime(num_workers: Optional[int], pin_memory: Optional[bool]):
    has_cuda = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 2 if has_cuda else 0
    if pin_memory is None:
        pin_memory = True if has_cuda else False
    return num_workers, pin_memory

def build_imagefolder_loaders_for_client(
    data_root: str,
    dataset_name: str,
    client_index: int,
    num_clients: int,
    partition_method: str = "iid",
    dirichlet_alpha: float = 0.5,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    client_test_ratio: float = 0.1,
    train_transform=None,
    test_transform=None,
):
    """
    通用 ImageFolder Loader:
      - data/<dataset_name>/train/
      - data/<dataset_name>/test/   (可选)
    """
    train_dir = os.path.join(data_root, dataset_name, "train")
    test_dir  = os.path.join(data_root, dataset_name, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    targets = [y for _, y in train_set.samples]
    if partition_method == "iid":
        mapping = iid_partition(targets, num_clients, seed)
    elif partition_method == "dirichlet":
        mapping = dirichlet_partition(targets, num_clients, dirichlet_alpha, seed)
    else:
        raise ValueError("partition_method must be 'iid' or 'dirichlet'")

    my_idx = mapping[client_index]
    train_idx, test_idx = split_train_test(my_idx, client_test_ratio, seed + client_index)

    train_subset = Subset(train_set, train_idx)

    # 本地测试集用 test_transform 重建一个 dataset
    test_set_for_client = datasets.ImageFolder(train_dir, transform=test_transform)
    test_subset = Subset(test_set_for_client, test_idx)

    # 公共测试集（如果有）
    public_test_loader = None
    if os.path.isdir(test_dir):
        public_set = datasets.ImageFolder(test_dir, transform=test_transform)
        nw, pm = _decide_loader_runtime(num_workers, pin_memory)
        public_test_loader = DataLoader(public_set, batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    nw, pm = _decide_loader_runtime(num_workers, pin_memory)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=nw, pin_memory=pm)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False, num_workers=nw, pin_memory=pm)

    return train_loader, test_loader, public_test_loader, len(train_subset), len(train_set.classes)