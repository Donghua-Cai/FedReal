# common/config.py
from dataclasses import dataclass, asdict

@dataclass
class FedConfig:
    # 基础训练配置
    num_clients: int = 3
    total_rounds: int = 30
    local_epochs: int = 5
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9

    # 数据划分
    # 可选: "iid" | "dirichlet" | "shards"
    partition_method: str = "iid"
    dirichlet_alpha: float = 0.5
    seed: int = 42
    sample_fraction: float = 1.0  # 每轮参与比例(0,1]

    # 模型与通信
    model_name: str = "resnet18"      # 客户端小模型
    max_message_mb: int = 128

    # 数据集信息
    num_classes: int = 10
    public_examples: int = 0          # 公共无标签集样本数，server_main 启动时填充

    # FedNew 相关（新增）
    group_num: int = 5                # 分组数量（按 label 分布聚类/轮转分组等）
    server_model_name: str = "resnet50"  # 服务器大模型用于 KD

    # 知识蒸馏超参（Server/Client 共用）
    server_kd_epochs: int = 3
    client_kd_epochs: int = 3
    kd_temperature: float = 1.0
    kd_alpha: float = 0.5             # L = (1-α) * CE + α * KL

    def to_dict(self):
        return asdict(self)