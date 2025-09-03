import random
import numpy as np
import torch
import logging
import sys

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_clients(all_client_ids, k, round_id, seed=42):
    # 可复现的按轮次采样
    rng = random.Random(seed + round_id)
    if k >= len(all_client_ids):
        return list(all_client_ids)
    return rng.sample(list(all_client_ids), k)

def setup_logger(name: str, level: int = logging.INFO, rank: int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:  # 避免重复添加 handler
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if rank is not None:
        logger = logging.LoggerAdapter(logger, {"rank": rank})

    return logger

def fmt_bytes(n: int) -> str:
    return f"{n} B ({n/1024/1024:.2f} MB)"