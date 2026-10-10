"""Hyperparameters and paths for the LDCT-IQAC ResNet18 baseline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_dir: Path = Path("./data/LDCTIQAG2023_train")
    checkpoint_path: Path = Path("./outputs/checkpoints/best_model.pth")
    output_dir: Path = Path("./outputs")

    # Data split (80/10/10, random because the dataset has no patient IDs)
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 50

    # Performance
    num_workers: int = 2
    seed: int = 42


CONFIG = Config()
