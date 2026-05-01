"""
Baseline flux optique pur : classification sur les 3 paires (flow_x, flow_y)
empilées en canaux -> 6 canaux d'entrée. Pas d'info d'apparence — uniquement
le mouvement.

Volontairement la même architecture que CNN_rgb (mêmes blocs, mêmes channels,
même classifier, même dropout) ; seule différence : 1ère conv à 6 canaux au
lieu de 3. Permet une comparaison apples-to-apples RGB vs flow.

Pendant : CNN_rgb.py (apparence pure).
Au-dessus : CNN_two_stream.py (combine les deux).
"""

import torch
import torch.nn as nn


INPUT_MODE = "flow"


class CNNFlow(nn.Module):
    def __init__(self, num_classes: int = 33, in_channels: int = 6, dropout: float = 0.5):
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
    return CNNFlow(num_classes=num_classes, in_channels=6)
