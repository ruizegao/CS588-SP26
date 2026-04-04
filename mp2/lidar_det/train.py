from __future__ import annotations

import random
import numpy as np
import torch



def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

__all__ = [
    "seed_everything",
    "resolve_device",
]
