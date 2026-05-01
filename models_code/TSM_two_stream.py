"""
Two-stream TSM (Temporal Shift Module, Lin et al. 2019).

Idée : au lieu d'une vraie convolution 3D, on décale gratuitement (zéro
paramètre, zéro FLOP) une fraction des canaux dans le temps avant chaque
bloc 2D. Ça injecte de l'information temporelle dans un réseau 2D classique.

Ici, deux branches indépendantes :
  - RGB  : 4 frames        (B, 4, 3, H, W) -> backbone TSM (T=4) -> (B, D)
  - Flow : 3 paires (x,y)  (B, 3, 2, H, W) -> backbone TSM (T=3) -> (B, D)
Fusion : concat des deux features -> Dropout -> Linear -> 33 logits
(mid-fusion, comme CNN_two_stream).

Backbone par modalité : strictement la même archi que CNN_rgb / CNN_flow
(blocs 32/64/128/256), avec un TemporalShift inséré en tête de chaque bloc.
Réduction temporelle finale : mean-pooling sur T (les features ont déjà été
mélangées temporellement par les 4 shifts).
"""

import torch
import torch.nn as nn


INPUT_MODE = "two_stream"


class TemporalShift(nn.Module):
    """
    TSM bidirectionnel offline. Sans paramètre, sans FLOP.

    Entrée:  (B*T, C, H, W) — T = num_segments (consécutif dans la dim batch).
    Sortie:  (B*T, C, H, W).
    Découpe les canaux en n_div parts :
      - 1ère 1/n_div : shift left  (canal au temps t reçoit la valeur au temps t+1)
      - 2ᵉ  1/n_div : shift right (canal au temps t reçoit la valeur au temps t-1)
      - reste        : inchangé
    Les bordures temporelles (t=0 pour right, t=T-1 pour left) sont remplies
    de zéros — comportement standard offline.
    """

    def __init__(self, num_segments: int, n_div: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nt, c, h, w = x.shape
        t = self.num_segments
        b = nt // t
        x = x.view(b, t, c, h, w)

        fold = c // self.n_div
        out = torch.zeros_like(x)
        # shift left (futur -> présent) sur la 1ère tranche
        out[:, :-1, :fold] = x[:, 1:, :fold]
        # shift right (passé -> présent) sur la 2ᵉ tranche
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        # reste inchangé
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class TSMBackbone(nn.Module):
    """
    Backbone 2D mêmes blocs que CNN_rgb/CNN_flow (32/64/128/256), mais avec
    un TemporalShift au début de chaque bloc. Réduit (B, T, C, H, W) à (B, D)
    par mean-pooling temporel après le pooling spatial.
    """

    def __init__(
        self,
        in_channels: int,
        num_segments: int,
        feature_dim: int = 256,
        n_div: int = 8,
    ):
        super().__init__()
        self.num_segments = num_segments

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                TemporalShift(num_segments, n_div),
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
        # (B, T, C, H, W) -> fold T dans le batch
        b, t, c, h, w = x.shape
        assert t == self.num_segments, f"attendu T={self.num_segments}, reçu T={t}"
        x = x.reshape(b * t, c, h, w)
        x = self.features(x)
        x = self.pool(x).flatten(1)        # (B*T, D)
        x = x.view(b, t, -1).mean(dim=1)   # (B, D), mean-pool temporel
        return x


class TwoStreamTSM(nn.Module):
    """
    Two-stream TSM : une backbone TSM par modalité, fusion mid-fusion.

    Entrées :
      frames : (B, 4, 3, H, W)
      flow   : (B, 3, 2, H, W)
    Sortie :
      logits : (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 33,
        num_frames: int = 4,
        num_flow_pairs: int = 3,
        feature_dim: int = 256,
        n_div: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.rgb_backbone = TSMBackbone(
            in_channels=3, num_segments=num_frames,
            feature_dim=feature_dim, n_div=n_div,
        )
        self.flow_backbone = TSMBackbone(
            in_channels=2, num_segments=num_flow_pairs,
            feature_dim=feature_dim, n_div=n_div,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(frames)    # (B, D)
        flow_feat = self.flow_backbone(flow)    # (B, D)
        return self.classifier(torch.cat([rgb_feat, flow_feat], dim=1))


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return TwoStreamTSM(num_classes=num_classes)
