from typing import Tuple
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_local(model, loader, epochs, optimizer, device="cpu") -> Tuple[float, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for ep in range(epochs):
        for x, y in tqdm(loader, desc=f"Train Epoch {ep+1}/{epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return (loss_sum / total, correct / total)


def evaluate(model, loader, device="cpu"):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return (loss_sum / total, correct / total)
