import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_clients(all_client_ids, k, round_id, seed=42):
    # 可复现的按轮次采样
    rng = random.Random(seed + round_id)
    if k >= len(all_client_ids):
        return list(all_client_ids)
    return rng.sample(list(all_client_ids), k)