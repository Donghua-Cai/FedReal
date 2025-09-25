import numpy as np
import torch

npz_path = "/home/caidonghua/FedReal/dataset/Cifar10/server.npz"

print(f"Loading {npz_path} ...")
obj = np.load(npz_path, allow_pickle=True)

print(f"Keys in npz: {list(obj.keys())}")

if "data" in obj:
    data = obj["data"].item()
    X_np, y_np = data["x"], data["y"]
elif "x" in obj and "y" in obj:
    X_np, y_np = obj["x"], obj["y"]
else:
    raise RuntimeError(f"Unexpected format: {list(obj.keys())}")

print(f"X shape: {X_np.shape}, dtype: {X_np.dtype}")
print(f"y shape: {y_np.shape}, dtype: {y_np.dtype}")

X = torch.as_tensor(X_np, dtype=torch.float32)
y = torch.as_tensor(y_np, dtype=torch.long)

print(f"Tensor X shape: {X.shape}, dtype: {X.dtype}")
print(f"Tensor y shape: {y.shape}, dtype: {y.dtype}")

for i in range(3):
    print(f"Sample {i}: X[{i}].shape={X[i].shape}, y[{i}].item()={y[i].item()}")

print("✅ NPZ file looks good, all samples can be accessed.")