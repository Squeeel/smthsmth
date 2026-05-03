"""
Two-stream TSM avec interleaving temporel et backbone partagé.

Idée : au lieu de deux backbones étanches, on construit une *unique* séquence
temporelle alternée RGB₀ - flow₀ - RGB₁ - flow₁ - RGB₂ - flow₂ - RGB₃ (T=7),
et un seul backbone partagé la traite. Comme TSM décale ±1 timestep, chaque
shift devient *naturellement* cross-modal : du RGB on regarde le flow voisin,
du flow on regarde le RGB voisin. La fusion est dense (à chaque bloc) et
quasiment gratuite.

Architecture :
  - Stem par modalité (2 convs 3×3 + BN + ReLU, no pool) : projette
    RGB(3 ch) et flow(2 ch) vers un espace partagé K canaux. Suffisamment
    profond pour homogénéiser les distributions sans dupliquer le backbone.
  - Interleave : assemble (B, 7, K, H, W) avec mod_idx [0,1,0,1,0,1,0].
  - Backbone partagé : 3 blocs (TSM(T=7) + 2 convs + BN-par-modalité +
    ReLU + MaxPool), tailles K → 64 → 128 → 256.
  - Pool : pool spatial, puis on sépare RGB-positions / flow-positions et
    on mean-pool chaque sous-séquence séparément (préserve l'asymétrie 4 vs 3).
  - Classifier : concat (rgb_feat, flow_feat) → Dropout → Linear → num_classes.

Pourquoi BN par modalité : RGB et flow ont des distributions très différentes
(intensités, gradients, sparsité). Une BN partagée moyennerait des stats
incompatibles et tuerait le bénéfice du backbone partagé. Deux BN routées
par mod_idx → chaque modalité garde ses stats, mais les *poids de conv*
restent partagés.

Diagnostic à faire après quelques epochs (cf. discussion stem) : probe
linéaire RGB-vs-flow sur la sortie du stem. Cible : ≤ 70% accuracy. Si
plus, le stem est trop superficiel — passer à 3 convs ou ajouter un bloc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


INPUT_MODE = "two_stream"


class TemporalShift(nn.Module):
    """
    TSM bidirectionnel offline. Sans paramètre, sans FLOP. Identique à
    celui de TSM_two_stream, mais ici num_segments = T_rgb + T_flow.

    Entrée:  (B*T, C, H, W).
    Sortie:  (B*T, C, H, W).
    Découpe les canaux en n_div parts :
      - 1ère 1/n_div : shift left  (canal au temps t reçoit la valeur en t+1)
      - 2ᵉ  1/n_div : shift right (canal au temps t reçoit la valeur en t-1)
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


class ModalityBatchNorm2d(nn.Module):
    """
    BN par modalité. Chaque sample de (B*T) est routé vers la BN
    correspondant à sa modalité (RGB ou flow), via un boolean mask.

    Les *poids de conv* restent partagés ; seules les stats (running mean/var)
    et les paramètres affines (γ, β) sont dédoublés. C'est ce qui permet à
    un backbone unique de digérer deux modalités à distributions différentes.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn_rgb = nn.BatchNorm2d(num_features)
        self.bn_flow = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor, mod_idx: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        rgb_mask = (mod_idx == 0)
        flow_mask = ~rgb_mask
        if rgb_mask.any():
            out[rgb_mask] = self.bn_rgb(x[rgb_mask])
        if flow_mask.any():
            out[flow_mask] = self.bn_flow(x[flow_mask])
        return out


class InterleavedBlock(nn.Module):
    """
    TSM (sur la séquence interleavée de longueur T_total) + 2 convs avec
    BN par modalité + ReLU + MaxPool. C'est ici que le shift temporel
    devient cross-modal : à chaque bloc, le décalage ±1 amène le voisin
    de l'autre modalité au présent.
    """

    def __init__(self, in_c: int, out_c: int, num_segments: int, n_div: int = 8):
        super().__init__()
        self.shift = TemporalShift(num_segments, n_div)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn1 = ModalityBatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.bn2 = ModalityBatchNorm2d(out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, mod_idx: torch.Tensor) -> torch.Tensor:
        x = self.shift(x)
        x = F.relu(self.bn1(self.conv1(x), mod_idx), inplace=True)
        x = F.relu(self.bn2(self.conv2(x), mod_idx), inplace=True)
        return self.pool(x)


class ModalityStem(nn.Module):
    """
    Stem spécifique par modalité. 2 convs 3×3 + BN + ReLU, sans pool.
    Projette l'entrée (3 ch RGB ou 2 ch flow) vers un espace partagé
    de K canaux à pleine résolution.

    Pas de pool ici pour garder les features stem à pleine résolution —
    facilite le diagnostic linéaire (probe RGB-vs-flow) et laisse les 3
    downsamples au backbone.
    """

    def __init__(self, in_c: int, k: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, k, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStreamTSMInterleaved(nn.Module):
    """
    Two-stream TSM avec interleaving temporel et backbone partagé.

    Entrées :
      frames : (B, T_rgb=4, 3, H, W)
      flow   : (B, T_flow=3, 2, H, W)
    Sortie :
      logits : (B, num_classes)

    Contrainte : T_flow == T_rgb - 1 (un flow entre chaque paire de frames
    consécutives), nécessaire pour le motif d'interleaving RGB_i / flow_i /
    RGB_{i+1}.
    """

    def __init__(
        self,
        num_classes: int = 33,
        num_frames: int = 4,
        num_flow_pairs: int = 3,
        stem_dim: int = 32,
        feature_dim: int = 256,
        n_div: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_flow_pairs != num_frames - 1:
            raise ValueError(
                f"interleaving suppose T_flow == T_rgb - 1 "
                f"(motif RGB_i / flow_i / RGB_{{i+1}}); "
                f"reçu T_rgb={num_frames}, T_flow={num_flow_pairs}"
            )
        self.t_rgb = num_frames
        self.t_flow = num_flow_pairs
        self.t_total = num_frames + num_flow_pairs   # 7 par défaut

        self.stem_rgb = ModalityStem(in_c=3, k=stem_dim)
        self.stem_flow = ModalityStem(in_c=2, k=stem_dim)

        # Indices des positions RGB (paires) et flow (impaires) dans la
        # séquence interleavée. Bufferisés pour suivre le device automatiquement.
        rgb_positions = [2 * i for i in range(self.t_rgb)]
        flow_positions = [2 * i + 1 for i in range(self.t_flow)]
        self.register_buffer("rgb_positions", torch.tensor(rgb_positions, dtype=torch.long))
        self.register_buffer("flow_positions", torch.tensor(flow_positions, dtype=torch.long))

        # mod_idx par position temporelle : 0 = RGB, 1 = flow.
        mod_idx_per_t = torch.zeros(self.t_total, dtype=torch.long)
        mod_idx_per_t[self.flow_positions] = 1
        self.register_buffer("mod_idx_per_t", mod_idx_per_t)

        self.block1 = InterleavedBlock(stem_dim, 64, self.t_total, n_div)
        self.block2 = InterleavedBlock(64, 128, self.t_total, n_div)
        self.block3 = InterleavedBlock(128, feature_dim, self.t_total, n_div)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, num_classes),
        )

    def _interleave(
        self, rgb_feat: torch.Tensor, flow_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        rgb_feat:  (B*T_rgb,  K, H, W)
        flow_feat: (B*T_flow, K, H, W)
        Retourne : (B*T_total, K, H, W) avec le motif RGB/flow alterné.
        """
        b_t_rgb, k, h, w = rgb_feat.shape
        b = b_t_rgb // self.t_rgb
        rgb_5d = rgb_feat.view(b, self.t_rgb, k, h, w)
        flow_5d = flow_feat.view(b, self.t_flow, k, h, w)

        out = torch.empty(
            b, self.t_total, k, h, w,
            device=rgb_feat.device, dtype=rgb_feat.dtype,
        )
        out[:, self.rgb_positions] = rgb_5d
        out[:, self.flow_positions] = flow_5d
        return out.reshape(b * self.t_total, k, h, w)

    def forward(self, frames: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, t_rgb, _, h, w = frames.shape
        _, t_flow, _, _, _ = flow.shape
        assert t_rgb == self.t_rgb and t_flow == self.t_flow, (
            f"attendu T_rgb={self.t_rgb}, T_flow={self.t_flow}; "
            f"reçu T_rgb={t_rgb}, T_flow={t_flow}"
        )

        # Stems spécifiques par modalité → espace K-canaux partagé
        rgb_feat = self.stem_rgb(frames.reshape(b * t_rgb, 3, h, w))
        flow_feat = self.stem_flow(flow.reshape(b * t_flow, 2, h, w))

        # Interleave en (B*T_total, K, H, W)
        x = self._interleave(rgb_feat, flow_feat)

        # mod_idx aligné avec x : pour chaque batch, on tile mod_idx_per_t.
        # Layout de x : [b0t0, b0t1, ..., b0t6, b1t0, ...] → repeat() match exactement.
        mod_idx = self.mod_idx_per_t.repeat(b)

        # Backbone partagé : TSM (cross-modal naturel) + per-modality BN
        x = self.block1(x, mod_idx)
        x = self.block2(x, mod_idx)
        x = self.block3(x, mod_idx)

        # Pool spatial → (B*T_total, D)
        x = self.spatial_pool(x).flatten(1)

        # Pool temporel séparé par modalité → préserve l'asymétrie 4/3
        x = x.view(b, self.t_total, -1)
        rgb_pooled = x[:, self.rgb_positions].mean(dim=1)    # (B, D)
        flow_pooled = x[:, self.flow_positions].mean(dim=1)  # (B, D)

        return self.classifier(torch.cat([rgb_pooled, flow_pooled], dim=1))


def build(num_classes: int = 33, **_) -> nn.Module:
    """Factory standard pour main.py."""
    return TwoStreamTSMInterleaved(num_classes=num_classes)
