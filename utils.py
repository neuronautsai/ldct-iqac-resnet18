"""Small helpers: seeding and metrics."""

import random

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, PLCC, SROCC and KROCC between predictions and MOS."""
    return {
        "N": len(labels),
        "MAE": float(np.mean(np.abs(preds - labels))),
        "RMSE": float(np.sqrt(np.mean((preds - labels) ** 2))),
        "PLCC": float(pearsonr(preds, labels)[0]),
        "SROCC": float(spearmanr(preds, labels)[0]),
        "KROCC": float(kendalltau(preds, labels)[0]),
    }
