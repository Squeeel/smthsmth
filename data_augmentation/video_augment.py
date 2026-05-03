"""
video_augment.py — augmentations spatiales partagées entre toutes les frames
RGB et toutes les cartes de flow d'une même vidéo.

Contexte : pour Something-Something, le signal utile est le **mouvement**.
Si on tire un crop ou une rotation différente pour chaque frame, on détruit la
cohérence temporelle et le réseau ne peut plus rien apprendre du déplacement.
torchvision.transforms ré-échantillonne ses paramètres à chaque appel — on ne
peut donc pas l'utiliser tel quel.

`VideoAugment` résout ça : on instancie un objet par vidéo, on tire les
paramètres une fois (`sample()`), puis `apply()` ré-applique les MÊMES
paramètres à chaque image — frames RGB et cartes flow confondues.

Patron d'intégration côté `SmthSmthDataset.__getitem__` :

    aug = VideoAugment(image_size=self.image_size)        # une instance par vidéo
    aug.sample()                                          # tire crop + angle
    # … puis dans _load_rgb / _load_gray, juste avant ToTensor :
    img = aug.apply(img, is_flow=False)   # ou True pour les flow_x / flow_y

Augmentations incluses :
  1. RandomResizedCrop : crop aléatoire (scale × ratio) ré-échantillonné à
     image_size × image_size.
  2. Petite rotation : angle aléatoire dans [-rotation_deg, +rotation_deg].

Volontairement absent (à ajouter plus tard, demandent une logique de label) :
  - flip horizontal / vertical (échangerait des classes : 018↔019, 008↔009)
  - reverse temporel       (échangerait : 010↔000 open/close, 003↔032 fold/unfold, …)
  - color jitter           (RGB only, à brancher en parallèle hors de cette classe)

Note sur la rotation des cartes de flow : on devrait théoriquement aussi
tourner le **vecteur** (mixage des composantes x et y). Pour rotation_deg ≤ 10°,
l'erreur introduite est négligeable (sin 10° ≈ 0.17) et toutes les impléms
courantes l'ignorent. Le `fill=128` au moment de la rotation préserve la
convention "vélocité nulle" sur les bords introduits par la rotation
(le flow uint8 est centré sur 128 ; cf. compute_optical_flow.py).
"""

from __future__ import annotations

import math
import random

from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode


class VideoAugment:
    """
    Augmentation spatiale déterministe sur une vidéo après appel à `sample()`.

    Une instance = une vidéo. Tirage des paramètres via `sample()`, puis
    `apply()` peut être appelé autant de fois qu'il y a d'images dans la vidéo
    (4 frames RGB + 6 cartes flow par exemple), toutes recevront la MÊME
    transformation spatiale.
    """

    def __init__(
        self,
        image_size: int = 112,
        scale: tuple[float, float] = (0.7, 1.0),
        ratio: tuple[float, float] = (0.8, 1.25),
        rotation_deg: float = 10.0,
    ):
        """
        image_size   : taille de sortie carrée (H = W = image_size).
        scale        : fraction d'aire conservée par le crop, tirée uniformément.
                       (0.7, 1.0) est conservateur — adapté à SS où un crop trop
                       agressif peut sortir l'action du cadre.
        ratio        : ratio largeur/hauteur du crop (avant resize).
        rotation_deg : amplitude max de rotation (en degrés, signe aléatoire).
        """
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.rotation_deg = rotation_deg

        self._angle: float | None = None
        self._crop: tuple[int, int, int, int] | None = None  # (top, left, h, w)

    # ------------------------------------------------------------------ sample

    def sample(self, source_size: tuple[int, int] | None = None) -> None:
        """
        Tire les paramètres aléatoires pour cette vidéo.

        source_size : (width, height) des images d'entrée. Si None, le crop
        sera tiré paresseusement à partir de la première image passée à
        `apply()` — pratique quand on ne connaît pas la taille à l'avance.
        Toutes les images d'une même vidéo doivent avoir la même taille
        (invariant respecté par les archives nettoyées).
        """
        self._angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        if source_size is not None:
            w, h = source_size
            self._crop = self._sample_crop(h, w)
        else:
            self._crop = None  # lazy

    def _sample_crop(self, h: int, w: int) -> tuple[int, int, int, int]:
        """
        Tire (top, left, h_crop, w_crop) selon (scale, ratio).
        Réplique torchvision.RandomResizedCrop.get_params sans dépendre de la
        signature interne (elle a bougé entre versions).
        """
        area = h * w
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * random.uniform(*self.scale)
            aspect = math.exp(random.uniform(*log_ratio))
            cw = int(round(math.sqrt(target_area * aspect)))
            ch = int(round(math.sqrt(target_area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
                return top, left, ch, cw

        # Fallback : crop centré le plus grand possible respectant `ratio`.
        in_ratio = w / h
        if in_ratio < min(self.ratio):
            cw = w
            ch = int(round(cw / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            ch = h
            cw = int(round(ch * max(self.ratio)))
        else:
            cw, ch = w, h
        top = (h - ch) // 2
        left = (w - cw) // 2
        return top, left, ch, cw

    # ------------------------------------------------------------------- apply

    def apply(self, img: Image.Image, *, is_flow: bool) -> Image.Image:
        """
        Applique la transfo déjà tirée à une image PIL.

        is_flow : True pour les cartes flow (mode "L" attendu). On utilise
        alors fill=128 pour la rotation — la convention "128 = vélocité nulle"
        est ainsi préservée sur les pixels créés aux coins par la rotation.
        Pour le RGB, fill=0 (noir).
        """
        if self._angle is None:
            raise RuntimeError("Appeler sample() avant apply().")

        if self._crop is None:
            w, h = img.size
            self._crop = self._sample_crop(h, w)
        top, left, ch, cw = self._crop

        # 1. Random resized crop : crop puis resize bilinéaire à image_size².
        img = F.resized_crop(
            img,
            top,
            left,
            ch,
            cw,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        # 2. Rotation. fill dépend de la modalité : 128 pour le flow (vélocité
        # nulle), 0 pour le RGB (noir). PIL "L" -> fill=[128], "RGB" -> fill=[0,0,0].
        if is_flow:
            fill = [128]
        else:
            fill = [0, 0, 0] if img.mode == "RGB" else [0]
        img = F.rotate(
            img,
            self._angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        )
        return img


# ----------------------------------------------------------------------- helper

def make_train_augment(image_size: int = 112) -> VideoAugment:
    """
    Construit un VideoAugment avec les valeurs par défaut adaptées au train.
    À n'utiliser que pour le split train — pas en val/test.
    """
    return VideoAugment(
        image_size=image_size,
        scale=(0.7, 1.0),
        ratio=(0.8, 1.25),
        rotation_deg=10.0,
    )
