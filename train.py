"""Train ResNet18 on LDCT-IQAC with MSE loss and keep the best validation checkpoint."""

import json
from dataclasses import asdict

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from data import load_master_list, make_loaders, split_master_list
from model import create_model
from utils import set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Train for one epoch and return the sample-weighted mean loss."""
    model.train()
    total = 0.0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        preds = model(imgs)
        loss = loss_fn(preds.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float]:
    """Return (mean loss, SROCC) on a loader."""
    model.eval()
    total = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).squeeze(1)
        total += loss_fn(preds, labels).item() * imgs.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return total / len(loader.dataset), float(spearmanr(preds_arr, labels_arr)[0])


def main() -> None:
    """Run the full training loop."""
    print("Device:", DEVICE)
    CONFIG.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(CONFIG.seed)
    df = load_master_list(CONFIG.data_dir)
    train_df, val_df, test_df = split_master_list(df, CONFIG.seed)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.to_csv(CONFIG.output_dir / f"{name}.csv", index=False)
    train_loader, val_loader, _ = make_loaders(train_df, val_df, test_df, CONFIG)

    set_seed(CONFIG.seed)
    model = create_model().to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_srocc": []}
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_srocc = evaluate(model, val_loader, loss_fn)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_srocc"].append(val_srocc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            config_dict = {k: str(v) for k, v in asdict(CONFIG).items()}
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config_dict,
                },
                CONFIG.checkpoint_path,
            )

        flag = " * best" if epoch == best_epoch else ""
        print(
            f"Epoch {epoch:02d}/{CONFIG.num_epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | val SROCC {val_srocc:.4f}{flag}"
        )

    with open(CONFIG.output_dir / "history.json", "w") as f:
        json.dump(history, f)
    print(f"Best epoch: {best_epoch} | val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
