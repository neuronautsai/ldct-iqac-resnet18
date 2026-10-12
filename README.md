# LDCT-IQAC ResNet18 Baseline

## Citation

**If you use this code, please cite:**

```bibtex
@misc{akgun2026ldctiqac,
  title={LDCT-IQAC ResNet18 Baseline: Image Quality Score Regression on Low-Dose CT Slices},
  author={Akg{\"u}n, Ceren},
  year={2026}
}
```

## Overview

A ResNet18 regression model that predicts the Mean Opinion Score (MOS, range 0 to 4) of a single 512x512 low-dose CT slice, trained on the LDCT-IQAC 2023 (MICCAI) challenge data.

## Features

- Transfer learning from ImageNet ResNet18 with a 1-channel input layer and a 1-output head
- MSE loss, Adam optimizer, best checkpoint chosen by validation loss
- Test metrics: MAE, RMSE, PLCC, SROCC, KROCC
- Training curves and predicted vs true MOS scatter plot

## Installation

```bash
git clone <repository-url>
cd ldct-iqac-resnet18

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install ruff pytest
```

## Dataset

**LDCT Image Quality Assessment Challenge, MICCAI 2023** ([challenge site](https://ldctiqac2023.grand-challenge.org/), [Zenodo record](https://zenodo.org/records/7833096))

- 1000 images in `.tif` format, all 512x512 float32 with values in [0, 1]
- MOS labels in `train.json`, range 0 to 4
- Expected layout: `data/LDCTIQAG2023_train/image/*.tif` and `data/LDCTIQAG2023_train/train.json`
- Train/Val/Test split: 800/100/100 images
- The split is random (seed 42) because the dataset has no patient IDs, so a patient-level split was not possible. Slices from the same patient may therefore appear in more than one split, which can make the scores optimistic.
- No resizing or rescaling is applied, the images are used as provided.

## Usage

### Training

```bash
python train.py
```

This writes the split CSVs, `history.json` and the best checkpoint (`outputs/checkpoints/best_model.pth`).

### Evaluation

```bash
python plot_curves.py
python test.py
```

`test.py` evaluates the best checkpoint once on the test split and saves metrics and the scatter plot to `outputs/`.

### Configuration

Edit `config.py` to modify hyperparameters:

```python
from dataclasses import dataclass


@dataclass
class Config:
    learning_rate: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 50
    num_workers: int = 2
    seed: int = 42
```

### Quality checks

```bash
make ci
```

## Architecture

`torchvision` ResNet18 initialised with ImageNet weights (`IMAGENET1K_V1`). The first convolution is replaced by a new randomly initialised `Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)` and the final layer by `Linear(512, 1)`. The model has 11,170,753 parameters.

## Results

Test set (N=100), checkpoint from epoch 35 (best validation loss 0.0488):

| Metric | Value |
| ------ | ----- |
| MAE | 0.1854 |
| RMSE | 0.2425 |
| PLCC | 0.9742 |
| SROCC | 0.9711 |
| KROCC | 0.8779 |

Results come from a single run with one random split, so they carry the caveat described in the Dataset section. Runs on other hardware or library versions may differ slightly.

![Training curves](assets/training_curves.png)

![Predicted vs true MOS on the test set](assets/scatter_test.png)

## Week 2: Option B (Loss & Training Track)

Experiments in `ldct_iqac_week2_optionB.ipynb` (Google Colab, Tesla T4). The notebook imports the modules of this repository and only changes the loss function or the learning rate scheduler. Model, data split (800/100/100), Adam lr 1e-4, batch size 16, 50 epochs and seed 42 are the same as in Week 1. For each run the checkpoint with the lowest validation loss is evaluated once on the test set (N = 100).

### B1: Loss functions

| Loss | Best epoch | MAE | RMSE | PLCC | SROCC | KROCC |
| ---- | ---------- | --- | ---- | ---- | ----- | ----- |
| MSE | 19 | 0.1932 | 0.2463 | 0.9724 | 0.9681 | 0.8688 |
| L1 (MAE) | 48 | 0.1984 | 0.2548 | 0.9729 | 0.9690 | 0.8750 |
| Huber (SmoothL1, beta = 1.0) | 43 | 0.1887 | 0.2328 | 0.9756 | 0.9703 | 0.8738 |
| Pairwise ranking (margin 0.1) | 14 | 2.1487 | 2.1695 | 0.9670 | 0.9714 | 0.8713 |
| Combined (0.5 MSE + 0.5 ranking) | 19 | 0.1916 | 0.2428 | 0.9728 | 0.9684 | 0.8700 |

The ranking loss only constrains the order of the predictions, not their scale, so its MAE and RMSE are large by design.

![B1 validation SROCC](assets/b1_val_srocc.png)

### B2: Learning rate schedulers (MSE loss)

| Scheduler | Best epoch | MAE | RMSE | PLCC | SROCC | KROCC |
| --------- | ---------- | --- | ---- | ---- | ----- | ----- |
| Constant lr (B1 MSE run) | 19 | 0.1932 | 0.2463 | 0.9724 | 0.9681 | 0.8688 |
| StepLR (step_size=10, gamma=0.1) | 47 | 0.1876 | 0.2352 | 0.9749 | 0.9709 | 0.8754 |
| CosineAnnealingLR (T_max=50, eta_min=1e-6) | 24 | 0.1890 | 0.2372 | 0.9742 | 0.9682 | 0.8700 |
| OneCycleLR (max_lr=1e-3, stepped per batch) | 39 | 0.1867 | 0.2363 | 0.9751 | 0.9713 | 0.8779 |

![B2 learning rate curves](assets/b2_lr_curves.png)

![B2 validation loss](assets/b2_val_loss.png)

### Notes

- The differences between the runs are small (SROCC within about 0.003 to 0.004) and of the same size as the epoch-to-epoch variation of the validation SROCC, so no method is declared the best.
- The MSE run in the notebook does not reproduce the Week 1 numbers exactly (epoch 19 and SROCC 0.9681 here, epoch 35 and SROCC 0.9711 in Week 1). The cause was not identified. The notebook run is used as the baseline for all Week 2 comparisons.
- Single run, one seed, one random split with no patient IDs, so there is no variance estimate. B3 (5-fold cross-validation) was not run.
- To reproduce, run the cells of the notebook in order on a GPU runtime. Result tables are in `assets/b1_table.csv` and `assets/b2_table.csv`.

## Project Structure

```
.
├── README.md
├── LICENSE (CC BY-NC 4.0)
├── Makefile
├── requirements.txt
├── pyproject.toml
├── config.py
├── data.py
├── model.py
├── train.py
├── test.py
├── plot_curves.py
├── utils.py
├── ldct_iqac_week2_optionB.ipynb
├── assets/
└── tests/
```

## References

- LDCT Image Quality Assessment Challenge, MICCAI 2023
- K. He et al., Deep Residual Learning for Image Recognition, CVPR 2016

## License

This project is licensed under CC BY-NC 4.0 (Creative Commons
Attribution-NonCommercial 4.0 International) - see the [LICENSE](LICENSE) file
for details.

## Contact

For questions, please open an issue in this repository or contact neuronautsai@gmail.com.
