import torch
import torch.nn as nn
from .models import resnet18, resnet50

def create_model(name: str = "resnet18", num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        return resnet18(num_classes)
    elif name == "resnet50":
        return resnet50(num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")