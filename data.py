"""Master list, 80/10/10 split and PyTorch dataset for LDCT-IQAC 2023."""

import json
from pathlib import Path

import imageio.v3 as iio
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import Config


def load_master_list(data_dir: Path) -> pd.DataFrame:
    """Build the (image_path, MOS) master list from train.json and image/."""
    with open(data_dir / "train.json") as f:
        data = json.load(f)
    df = pd.DataFrame([{"Filename": k, "MOS Score": v} for k, v in data.items()])
    df["image_path"] = df["Filename"].apply(lambda x: str(data_dir / "image" / x))
    return df


def split_master_list(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random 80/10/10 split. The dataset has no patient IDs, so no patient-level split."""
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)
    return train_df, val_df, test_df


class LDCTIQADataset(Dataset):
    """Returns (image, MOS) with image of shape (1, H, W) in [0, 1]."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.paths = df["image_path"].tolist()
        self.scores = df["MOS Score"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Images are already float32 in [0, 1], so no extra scaling is applied
        image = iio.imread(self.paths[idx]).copy()
        image = torch.from_numpy(image).float().unsqueeze(0)
        score = torch.tensor(self.scores[idx], dtype=torch.float32)
        return image, score


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train (shuffled), val and test DataLoaders."""
    kwargs = {"batch_size": config.batch_size, "num_workers": config.num_workers}
    train_loader = DataLoader(LDCTIQADataset(train_df), shuffle=True, **kwargs)
    val_loader = DataLoader(LDCTIQADataset(val_df), shuffle=False, **kwargs)
    test_loader = DataLoader(LDCTIQADataset(test_df), shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
