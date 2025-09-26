from dataclasses import dataclass

@dataclass
class FedConfig:
    num_clients: int = 3
    total_rounds: int = 30
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9
    seed: int = 42
    sample_fraction: float = 1.0
    model_name: str = "resnet18"
    feature_dim: int =512
    max_message_mb: int = 128
    num_classes: int = 10
    encoder_ratio: float = 1.0

    def to_dict(self):
        return self.__dict__.copy()