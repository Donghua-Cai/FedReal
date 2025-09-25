import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "server_data_loader",
    "client_data_loader",
]

class NpzDataset(Dataset):
    """Dataset for loading npz files with x (features) and y (labels)."""

    def __init__(self, X_np, y_np):
        # 转成 torch.Tensor 存在内存里
        self.X = torch.as_tensor(X_np, dtype=torch.float32)
        self.y = torch.as_tensor(y_np, dtype=torch.long)

        if len(self.X) != len(self.y):
            raise ValueError(f"X and y have different lengths: {len(self.X)} vs {len(self.y)}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _load_npz_xy(npz_path: str):
    obj = np.load(npz_path, allow_pickle=True)

    if "data" in obj:  # client: data -> {x,y}
        data_field = obj["data"]
        data = data_field.item() if hasattr(data_field, "item") else data_field.tolist()
        if not isinstance(data, dict) or "x" not in data or "y" not in data:
            raise ValueError(f"'data' must be a dict with keys 'x' and 'y' in {npz_path}")
        X_np, y_np = data["x"], data["y"]

    elif "x" in obj and "y" in obj:  # server: top-level x,y
        X_np, y_np = obj["x"], obj["y"]

    else:
        raise KeyError(
            f"Expected either 'data' (with x,y) or top-level 'x'/'y' in {npz_path}. "
            f"Found keys: {list(obj.keys())}"
        )

    return NpzDataset(X_np, y_np)


def server_data_loader(
    data_root: str, dataset_name: str, batch_size: int,
    shuffle: bool = False, num_workers: int = 0
):
    """Return DataLoader for server's public test dataset."""
    server_data_file = os.path.join(data_root, dataset_name, "server.npz")
    server_dataset = _load_npz_xy(server_data_file)
    return DataLoader(
        server_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def client_data_loader(
    data_root: str, dataset_name: str, cid: int, batch_size: int,
    train_shuffle: bool = False, test_shuffle: bool = False,
    num_workers: int = 0
):
    """Return DataLoader for one client (train + test)."""
    train_path = os.path.join(data_root, dataset_name, "train", f"{cid}.npz")
    test_path = os.path.join(data_root, dataset_name, "test", f"{cid}.npz")

    train_dataset = _load_npz_xy(train_path)
    test_dataset = _load_npz_xy(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader