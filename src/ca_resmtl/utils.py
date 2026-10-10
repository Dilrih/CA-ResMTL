"""Random seeds and numerical checks."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def finite_check(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"[NaN/Inf] {name} contains non-finite values.")
