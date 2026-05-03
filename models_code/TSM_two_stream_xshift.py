"""
Two-stream TSM avec cross-modal shift (variante de TSM_two_stream).

Reprend exactement l'archi de TSM_two_stream (mêmes blocs 32/64/128/256, même
TemporalShift en tête de chaque bloc) et ajoute un *cross-modal shift* entre
chaque paire de blocs : on échange une petite fraction des canaux entre la
branche RGB et la branche flow au timestep aligné. Zéro paramètre, zéro FLOP
en plus — exactement la philosophie TSM, étendue à la fusion inter-modalité.

But : forcer la fusion apparence + mouvement *bien avant* le classifier final.
Dans TSM_two_stream, les deux branches sont étanches jusqu'au `cat` final
(late fusion déguisée en mid-fusion). Ici, l'info circule entre les deux
streams à 3 profondeurs.

Alignement temporel :
  flow_i correspond à la transition entre frame_i et frame_{i+1}, donc on
  aligne flow_i ↔ frame_i. Avec T_rgb=4 et T_flow=3, frame_3 n'a pas de
  partenaire flow et reste inchangée aux points d'échange.

Choix des canaux :
  TemporalShift opère sur [:fold] et [fold:2*fold]. CrossModalShift opère sur
  des slices disjoints ([2*fold:3*fold] et [3*fold:4*fold]) pour ne pas
  écraser ce que TSM vient d'écrire. n_div >= 4 requis (par défaut 8 → 1/8
  des canaux par direction d'échange, 1/4 du total touché par cross-modal).
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
    Bordures temporelles remplies de zéros.
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
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
        return out.view(nt, c, h, w)


class CrossModalShift(nn.Module):
    """
    Échange une petite fraction des canaux entre RGB et flow, au timestep
    aligné. Zéro paramètre, zéro FLOP (slicing/copie uniquement).

    Entrées:
        rgb:  (B*T_rgb,  C, H, W)
        flow: (B*T_flow, C, H, W)
    Sorties: (rgb_out, flow_out) mêmes shapes.

    Canaux échangés (disjoints des canaux utilisés par TemporalShift):
      - [2*fold : 3*fold] : RGB reçoit le contenu flow au timestep aligné
      - [3*fold : 4*fold] : flow reçoit le contenu RGB au timestep aligné
      - reste              : inchangé
    où fold = C // n_div. Les timesteps RGB sans partenaire flow
    (frame_3 quand T_rgb=4, T_flow=3) restent inchangés.
    """

    def __init__(self, t_rgb: int, t_flow: int, n_div: int = 8):
        super().__init__()
        if n_div < 4:
            raise ValueError(
                f"n_div doit être >= 4 (reçu {n_div}) pour laisser de la place "
                f"après les ranges utilisés par TSM"
            )
        self.t_rgb = t_rgb
        self.t_flow = t_flow
        self.t_common = min(t_rgb, t_flow)
        self.n_div = n_div

    def forward(
        self, rgb: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nt_rgb, c, h, w = rgb.shape
        nt_flow = flow.shape[0]
        b = nt_rgb // self.t_rgb
        rgb_5d = rgb.view(b, self.t_rgb, c, h, w)
        flow_5d = flow.view(b, self.t_flow, c, h, w)

        fold = c // self.n_div
        tc = self.t_common
        out_rgb = rgb_5d.clone()
        out_flow = flow_5d.clone()

        out_rgb[:, :tc, 2 * fold:3 * fold] = flow_5d[:, :tc, 2 * fold:3 * fold]
        out_flow[:, :tc, 3 * fold:4 * fold] = rgb_5d[:, :tc, 3 * fold:4 * fold]

        return out_rgb.view(nt_rgb, c, h, w), out_flow.view(nt_flow, c, h, w)


class TSMBackbone(nn.Module):
    """
    Backbone 2D blocs 32/64/128/256 avec TemporalShift en tête de chaque bloc.
    Blocs exposés en `nn.ModuleList` pour permettre l'orchestration externe
    (cross-modal shift entre blocs). Réduit (B, T, C, H, W) à (B, D) par
    mean-pool temporel après le pool spatial.
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

        self.blocks = nn.ModuleList([
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, feature_dim),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def fold_time(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        b, t, c, h, w = x.shape
        assert t == self.num_segments, f"attendu T={self.num_segments}, reçu T={t}"
        return x.reshape(b * t, c, h, w), b

    def unfold_pool(self, x: torch.Tensor, b: int) -> torch.Tensor:
        x = self.pool(x).flatten(1)                          # (B*T, D)
        return x.view(b, self.num_segments, -1).mean(dim=1)  # (B, D)


class TwoStreamTSMXShift(nn.Module):
    """
    Two-stream TSM avec cross-modal shift entre les blocs.

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
        cross_n_div: int = 8,
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
        # Un échange après chaque bloc sauf le dernier (on pool ensuite).
        self.cross_shifts = nn.ModuleList([
            CrossModalShift(num_frames, num_flow_pairs, n_div=cross_n_div)
            for _ in range(len(self.rgb_backbone.blocks) - 1)
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, num_classes),
        )

    def forward(self, frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        rgb, b = self.rgb_backbone.fold_time(frames)
        flw, _ = self.flow_backbone.fold_time(flow)

        rgb_blocks = self.rgb_backbone.blocks
        flw_blocks = self.flow_backbone.blocks
        last = len(rgb_blocks) - 1
        for i, (rgb_block, flw_block) in enumerate(zip(rgb_blocks, flw_blocks)):
            rgb = rgb_block(rgb)
            flw = flw_block(flw)
            if i < last:
                rgb, flw = self.cross_shifts[i](rgb, flw)

        rgb_feat = self.rgb_backbone.unfold_pool(rgb, b)
        flw_feat = self.flow_backbone.unfold_pool(flw, b)
        return self.classifier(torch.cat([rgb_feat, flw_feat], dim=1))


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return TwoStreamTSMXShift(num_classes=num_classes)
