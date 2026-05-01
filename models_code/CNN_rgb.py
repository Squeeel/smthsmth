"""
Baseline RGB pur : classification sur la PREMIÈRE frame uniquement.
Pas d'info temporelle, pas de mouvement — apparence brute.

Pendant : CNN_flow.py (même archi, branche flux pure).
Au-dessus : CNN_two_stream.py (combine les deux).
"""

import torch
import torch.nn as nn


INPUT_MODE = "rgb_first"


class CNNRgb(nn.Module):
    def __init__(self, num_classes: int = 33, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            self._conv_block(in_channels, 32),   # 1
            self._conv_block(32, 64),            # 2
            self._conv_block(64, 128),           # 3
            self._conv_block(128, 256),          # 4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return CNNRgb(num_classes=num_classes, in_channels=3)
