import torch
import torch.nn as nn
from torchvision import models

def resnet18(num_classes: int = 10):
    model = models.resnet18(weights=None)
    # 将最后一层替换为 num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet50(num_classes: int = 10):
    model = models.resnet50(weights=None)
    # 将最后一层替换为 num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model