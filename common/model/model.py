import torch
import torch.nn as nn
from torchvision import models


def create_model(name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        # 将最后一层替换为 num_classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")