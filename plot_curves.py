"""Plot training curves (train/val loss and validation SROCC) from history.json."""

import json

import matplotlib.pyplot as plt
import numpy as np

from config import CONFIG


def main() -> None:
    """Save training_curves.png with the best epoch marked."""
    with open(CONFIG.output_dir / "history.json") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = int(np.argmin(history["val_loss"])) + 1

    fig, (loss_ax, metric_ax) = plt.subplots(1, 2, figsize=(12, 4))
    loss_ax.plot(epochs, history["train_loss"], label="train")
    loss_ax.plot(epochs, history["val_loss"], label="val")
    loss_ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best (epoch {best_epoch})")
    loss_ax.set(xlabel="Epoch", ylabel="MSE loss", title="Loss")
    loss_ax.legend()

    metric_ax.plot(epochs, history["val_srocc"], color="tab:green")
    metric_ax.axvline(best_epoch, color="gray", linestyle="--")
    metric_ax.set(xlabel="Epoch", ylabel="Validation SROCC", title="Validation SROCC")

    fig.tight_layout()
    fig.savefig(CONFIG.output_dir / "training_curves.png", dpi=200)
    print("Best epoch:", best_epoch)


if __name__ == "__main__":
    main()
