"""ResNet18 regression model for single-channel CT slices."""

from torch import nn
from torchvision import models


def create_model(pretrained: bool = True) -> nn.Module:
    """ResNet18 with a 1-channel input conv and a single-output head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # New randomly initialised conv1: the pretrained RGB conv1 weights are not reused
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
