"""
Two-stream R(2+1)D — décomposition d'une convolution 3D `k×k×k` en :
    Conv3D spatial (1, k, k) + BN + ReLU + Conv3D temporal (k, 1, 1) + BN + ReLU

Avantages vs pur 3D pour ce dataset (T=4 RGB, T=3 flow) :
  - Moins de params : k²+k au lieu de k³ par "unit" (à channels constants)
  - Non-linéarité supplémentaire entre spatial et temporel → expressivité OK
  - Optim plus stable (cf. Tran et al. 2018, "A Closer Look at Spatiotemporal
    Convolutions for Action Recognition")
  - Moins enclin à l'overfit, ce qu'on cherche ici

Archi : 4 blocs (32/64/128/256), un seul (2+1)D unit par bloc + MaxPool spatial.
Une backbone par modalité (RGB T=4, flow T=3). Fusion par concat des features
poolées (spatial + temporel) puis Linear -> 33 logits. Cohérent avec
CNN_two_stream et TSM_two_stream.
"""

import torch
import torch.nn as nn


INPUT_MODE = "two_stream"


class Conv2plus1D(nn.Module):
    """
    Unité (2+1)D : conv spatiale 2D puis conv temporelle 1D, séparées par
    BN + ReLU. Entrée / sortie : (B, C, T, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.spatial = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, pad, pad), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.temporal = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(pad, 0, 0), bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.spatial(x)))
        x = self.act2(self.bn2(self.temporal(x)))
        return x


class R2plus1DBackbone(nn.Module):
    """
    Backbone (2+1)D 4 blocs (32/64/128/256). Pool final adaptatif sur
    (T, H, W) -> (B, feature_dim).

    Entrée: (B, T, C_in, H, W) — comme retourné par SmthSmthDataset.
    Sortie: (B, feature_dim).
    """

    def __init__(self, in_channels: int, feature_dim: int = 256):
        super().__init__()
        # MaxPool spatial seulement (T petit -> on ne le réduit pas).
        spatial_pool = lambda: nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.features = nn.Sequential(
            Conv2plus1D(in_channels, 32),
            spatial_pool(),
            Conv2plus1D(32, 64),
            spatial_pool(),
            Conv2plus1D(64, 128),
            spatial_pool(),
            Conv2plus1D(128, feature_dim),
            spatial_pool(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)  # collapse (T, H, W) -> 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C, H, W) -> (B, C, T, H, W) pour Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class TwoStreamR2plus1D(nn.Module):
    """
    Two-stream (2+1)D : une backbone par modalité, fusion mid-fusion.

    Entrées :
      frames : (B, 4, 3, H, W)
      flow   : (B, 3, 2, H, W)
    Sortie :
      logits : (B, num_classes)
    """

    def __init__(self, num_classes: int = 33, feature_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.rgb_backbone = R2plus1DBackbone(in_channels=3, feature_dim=feature_dim)
        self.flow_backbone = R2plus1DBackbone(in_channels=2, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(frames)
        flow_feat = self.flow_backbone(flow)
        return self.classifier(torch.cat([rgb_feat, flow_feat], dim=1))


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return TwoStreamR2plus1D(num_classes=num_classes)
