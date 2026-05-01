"""
CNN two-stream à poids partagés.

- Branche RGB  : chaque frame passe indépendamment dans le MÊME CNN (poids
                 partagés) — N_frames vecteurs de features.
- Branche flow : pour chaque pas de temps, flow_x et flow_y sont empilés en
                 2 canaux puis passés dans le MÊME CNN flow (poids partagés) —
                 N_pairs vecteurs de features.
- Fusion       : tous les vecteurs sont concaténés -> une couche dense -> logits.
"""

import torch
import torch.nn as nn


INPUT_MODE = "two_stream"


class _SharedCNN(nn.Module):
    """Backbone CNN partagé : (B, C, H, W) -> (B, feature_dim)."""

    def __init__(self, in_channels: int, feature_dim: int = 256):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, feature_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class TwoStreamSharedCNN(nn.Module):
    """
    Forward:
      frames : (B, N_frames=4, 3, H, W)
      flow   : (B, N_pairs=3,  2, H, W)   # flow_x et flow_y empilés sur l'axe canal
    Sortie:
      logits : (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 33,
        num_frames: int = 4,
        num_flow_pairs: int = 3,
        feature_dim: int = 256,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.num_flow_pairs = num_flow_pairs

        # Un seul CNN par modalité = poids partagés sur tous les pas de temps
        self.rgb_cnn = _SharedCNN(in_channels=3, feature_dim=feature_dim)
        self.flow_cnn = _SharedCNN(in_channels=2, feature_dim=feature_dim)

        total_features = (num_frames + num_flow_pairs) * feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(total_features, num_classes),
        )

    def forward(self, frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # --- RGB : (B, T_rgb, 3, H, W) -> fold T dans le batch -> CNN partagé
        B, T_rgb, C_rgb, H, W = frames.shape
        rgb_feats = self.rgb_cnn(frames.reshape(B * T_rgb, C_rgb, H, W))
        rgb_feats = rgb_feats.reshape(B, T_rgb * rgb_feats.size(1))   # (B, T_rgb * D)

        # --- Flow : (B, T_flow, 2, H, W) -> fold T dans le batch -> CNN partagé
        B, T_flow, C_flow, H, W = flow.shape
        flow_feats = self.flow_cnn(flow.reshape(B * T_flow, C_flow, H, W))
        flow_feats = flow_feats.reshape(B, T_flow * flow_feats.size(1))  # (B, T_flow * D)

        # --- Concaténation puis couche dense
        combined = torch.cat([rgb_feats, flow_feats], dim=1)
        return self.classifier(combined)


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return TwoStreamSharedCNN(num_classes=num_classes)
