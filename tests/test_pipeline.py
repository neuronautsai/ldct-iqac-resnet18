"""Smoke tests on synthetic data (no real LDCT images needed)."""

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from data import LDCTIQADataset, load_master_list, split_master_list
from model import create_model
from utils import compute_metrics


@pytest.fixture()
def fake_data_dir(tmp_path: Path) -> Path:
    (tmp_path / "image").mkdir()
    scores = {}
    rng = np.random.default_rng(0)
    for i in range(20):
        name = f"{i:04d}.tif"
        iio.imwrite(tmp_path / "image" / name, rng.random((64, 64), dtype=np.float32))
        scores[name] = float(i % 5)
    with open(tmp_path / "train.json", "w") as f:
        json.dump(scores, f)
    return tmp_path


def test_split_sizes_and_no_overlap(fake_data_dir: Path) -> None:
    df = load_master_list(fake_data_dir)
    train_df, val_df, test_df = split_master_list(df, seed=42)
    assert (len(train_df), len(val_df), len(test_df)) == (16, 2, 2)
    names = [set(d["Filename"]) for d in (train_df, val_df, test_df)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])


def test_dataset_item_shapes(fake_data_dir: Path) -> None:
    df = load_master_list(fake_data_dir)
    image, score = LDCTIQADataset(df)[0]
    assert image.shape == (1, 64, 64) and image.dtype == torch.float32
    assert score.dtype == torch.float32


def test_model_output_shape() -> None:
    model = create_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 1)


def test_compute_metrics_perfect_prediction() -> None:
    labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(labels.copy(), labels)
    assert metrics["MAE"] == 0.0 and metrics["RMSE"] == 0.0
    assert metrics["PLCC"] == pytest.approx(1.0)
    assert metrics["SROCC"] == pytest.approx(1.0)
    assert metrics["KROCC"] == pytest.approx(1.0)
