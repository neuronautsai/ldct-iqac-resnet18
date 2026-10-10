"""Evaluate the best checkpoint once on the held-out test split."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import CONFIG
from data import make_loaders
from model import create_model
from utils import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(model: torch.nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, targets) over a loader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, scores in tqdm(loader, desc="Test"):
            out = model(images.to(DEVICE)).squeeze(1).cpu().numpy()
            preds.append(out)
            targets.append(scores.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    """Load the best checkpoint, compute test metrics and save the scatter plot."""
    train_df = pd.read_csv(CONFIG.output_dir / "train.csv")
    val_df = pd.read_csv(CONFIG.output_dir / "val.csv")
    test_df = pd.read_csv(CONFIG.output_dir / "test.csv")
    _, _, test_loader = make_loaders(train_df, val_df, test_df, CONFIG)

    ckpt = torch.load(CONFIG.checkpoint_path, map_location=DEVICE)
    model = create_model(pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded epoch: {ckpt['epoch']}, val loss: {ckpt['val_loss']:.4f}")

    preds, targets = predict(model, test_loader)
    metrics = compute_metrics(preds, targets)
    for key, value in metrics.items():
        print(f"{key}: {value}" if key == "N" else f"{key}: {value:.4f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets, preds, alpha=0.7)
    ax.plot([0, 4], [0, 4], "k--", label="y = x")
    ax.set_xlabel("True MOS")
    ax.set_ylabel("Predicted MOS")
    ax.set_title(f"Test set (N={metrics['N']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CONFIG.output_dir / "scatter_test.png", dpi=200)

    with open(CONFIG.output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(CONFIG.output_dir / "test_preds.npy", preds)
    np.save(CONFIG.output_dir / "test_targets.npy", targets)


if __name__ == "__main__":
    main()
