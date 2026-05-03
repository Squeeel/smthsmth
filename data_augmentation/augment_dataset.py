"""
augment_dataset.py — Génère un dataset augmenté hors-ligne, puis empaquette les
archives tar correspondantes.

Pour chaque vidéo du split (par défaut `train`) :
  1. Instancie un `VideoAugment` et tire ses paramètres une seule fois (cf.
     video_augment.py — la cohérence temporelle exige le MÊME crop / la MÊME
     rotation pour toutes les frames RGB et toutes les cartes flow d'une vidéo).
  2. Applique cette transfo aux 4 frames RGB ET aux 6 cartes flow (3 paires
     flow_x / flow_y), avec `is_flow=True/False` pour gérer correctement le
     `fill` (128 pour le flow, 0 pour le RGB).
  3. Sauvegarde en miroir sous `data/frames_augmented/` et `data/optical_flow_augmented/`.

Le nombre de copies par vidéo est paramétrable (`--copies`). Chaque copie reçoit
un suffixe `_aug{n}` dans le nom de dossier vidéo (ex. `video_10061_aug0`) afin
que les archives augmentées puissent coexister avec les originales sans collision
si on les fusionne plus tard.

Une fois la génération terminée, le script empaquette les nouvelles données via
la même routine que `pack_dataset.py` (réutilisée telle quelle), produisant :
    data/archives/frames_{split}_aug.tar  + .index.json
    data/archives/flow_{split}_aug.tar    + .index.json

Usage (depuis n'importe où — les chemins par défaut sont résolus relativement
à la racine du projet, pas au CWD) :
    uv run data_augmentation/augment_dataset.py                       # train, 1 copie
    uv run data_augmentation/augment_dataset.py --copies 2 --workers 8
    uv run data_augmentation/augment_dataset.py --splits train val    # plusieurs splits
    uv run data_augmentation/augment_dataset.py --no-pack             # juste les images
    uv run data_augmentation/augment_dataset.py --image-size 224 --seed 42

Notes :
  - Par défaut, le split `test` est ignoré (pas de labels, pas d'intérêt à
    augmenter ce qu'on évalue). Possible via `--splits test` si on veut.
  - La taille de sortie reste celle des originaux par défaut (224 px) pour
    rester compatible avec le pipeline d'entraînement existant qui fait son
    propre resize à `image_size`. À forcer plus petit pour gagner de la place
    si on veut.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Le script vit dans data_augmentation/ mais doit pouvoir importer pack_dataset
# qui est à la racine du projet. On ajoute le parent au sys.path AVANT les
# imports locaux. video_augment.py est dans le même dossier — donc trouvé via
# sys.path[0] (ajouté automatiquement par Python pour le dossier du script).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pack_dataset import pack_split  # noqa: E402  (à la racine du projet)
from video_augment import VideoAugment  # noqa: E402  (même dossier)


SPLITS = ("train", "val", "test")
EXPECTED_FRAMES = [f"frame_{i:03d}.jpg" for i in range(4)]
EXPECTED_FLOW = [f"flow_{c}_{i:03d}.jpg" for i in range(3) for c in ("x", "y")]
JPEG_QUALITY = 95  # aligné sur le défaut de cv2.imwrite (compute_optical_flow.py)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_videos(frames_root: Path, split: str) -> list[tuple[str, Path]]:
    """
    Retourne [(rel_parent, video_dir), ...] où :
      - rel_parent : chemin relatif du *parent* de la vidéo depuis frames_root
                     (ex. "train/000_Closing_something" pour train/val,
                      "test" pour test).
      - video_dir  : chemin absolu vers le dossier vidéo source.
    """
    split_dir = frames_root / split
    if not split_dir.is_dir():
        return []

    videos: list[tuple[str, Path]] = []
    if split == "test":
        for v in sorted(split_dir.iterdir()):
            if v.is_dir():
                videos.append((split, v))
    else:
        for c in sorted(split_dir.iterdir()):
            if not c.is_dir():
                continue
            rel_parent = f"{split}/{c.name}"
            for v in sorted(c.iterdir()):
                if v.is_dir():
                    videos.append((rel_parent, v))
    return videos


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def augment_one_video(args: tuple) -> str | None:
    """
    Augmente une vidéo (toutes ses copies) et écrit le résultat sur disque.
    Lance dans un sous-process — `args` doit être picklable.

    args = (
        video_dir          : Path     dossier source des frames
        flow_video_dir     : Path     dossier source des flows
        out_frames_parent  : Path     ex. data/frames_augmented/train/000_Closing_something
        out_flow_parent    : Path     ex. data/optical_flow_augmented/train/000_Closing_something
        copies             : int      nb de versions augmentées à produire
        image_size         : int      taille de sortie carrée
        scale              : tuple[float,float]
        ratio              : tuple[float,float]
        rotation_deg       : float
        seed               : int | None    seed reproductible pour CETTE vidéo
    )

    Retourne None si OK, sinon une chaîne d'erreur.
    """
    (video_dir, flow_video_dir, out_frames_parent, out_flow_parent,
     copies, image_size, scale, ratio, rotation_deg, seed) = args

    try:
        # Vérifie sources complètes — invariant après clean_dataset.
        for fname in EXPECTED_FRAMES:
            if not (video_dir / fname).is_file():
                return f"frames manquantes : {video_dir}/{fname}"
        for fname in EXPECTED_FLOW:
            if not (flow_video_dir / fname).is_file():
                return f"flow manquant : {flow_video_dir}/{fname}"

        rng = random.Random(seed) if seed is not None else random.Random()

        for n in range(copies):
            aug = VideoAugment(
                image_size=image_size,
                scale=scale,
                ratio=ratio,
                rotation_deg=rotation_deg,
            )
            # Sub-seed déterministe par copie pour pouvoir rejouer.
            sub_seed = rng.randrange(2**31)
            # VideoAugment utilise `random` global — on l'amorce localement.
            random.seed(sub_seed)
            aug.sample()

            video_name = f"{video_dir.name}_aug{n}"
            out_v_frames = out_frames_parent / video_name
            out_v_flow = out_flow_parent / video_name
            out_v_frames.mkdir(parents=True, exist_ok=True)
            out_v_flow.mkdir(parents=True, exist_ok=True)

            # 4 frames RGB.
            for fname in EXPECTED_FRAMES:
                with Image.open(video_dir / fname) as img:
                    img = img.convert("RGB")
                    out = aug.apply(img, is_flow=False)
                    out.save(out_v_frames / fname, quality=JPEG_QUALITY)

            # 6 cartes flow (mode "L").
            for fname in EXPECTED_FLOW:
                with Image.open(flow_video_dir / fname) as img:
                    img = img.convert("L")
                    out = aug.apply(img, is_flow=True)
                    out.save(out_v_flow / fname, quality=JPEG_QUALITY)

        return None
    except Exception as e:
        return f"{video_dir} : {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def augment_split(
    split: str,
    frames_root: Path,
    flow_root: Path,
    out_frames_root: Path,
    out_flow_root: Path,
    copies: int,
    image_size: int,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    rotation_deg: float,
    workers: int,
    seed: int | None,
) -> tuple[int, list[str]]:
    """
    Augmente toutes les vidéos d'un split. Retourne (n_videos_traitées, erreurs).
    """
    videos = list_videos(frames_root, split)
    if not videos:
        print(f"  [skip] {split} : aucune vidéo sous {frames_root / split}")
        return 0, []

    print(f"  {split} : {len(videos)} vidéo(s) × {copies} copie(s) = "
          f"{len(videos) * copies} sortie(s)")

    # Tâches : on dérive un seed déterministe par vidéo à partir du seed global,
    # ce qui rend le run reproductible indépendamment de l'ordre des workers.
    base_rng = random.Random(seed) if seed is not None else None
    tasks = []
    for rel_parent, video_dir in videos:
        flow_video_dir = flow_root / rel_parent / video_dir.name
        out_frames_parent = out_frames_root / rel_parent
        out_flow_parent = out_flow_root / rel_parent
        video_seed = base_rng.randrange(2**31) if base_rng is not None else None
        tasks.append((
            video_dir, flow_video_dir,
            out_frames_parent, out_flow_parent,
            copies, image_size, scale, ratio, rotation_deg, video_seed,
        ))

    errors: list[str] = []
    if workers <= 1:
        for t in tqdm(tasks, desc=f"augment {split}", unit="video"):
            res = augment_one_video(t)
            if res is not None:
                errors.append(res)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in tqdm(
                ex.map(augment_one_video, tasks, chunksize=8),
                total=len(tasks),
                desc=f"augment {split}",
                unit="video",
            ):
                if res is not None:
                    errors.append(res)

    return len(tasks), errors


def pack_augmented(
    split: str,
    out_frames_root: Path,
    out_flow_root: Path,
    archives_root: Path,
) -> None:
    """
    Empaquette le split augmenté via pack_dataset.pack_split (même logique que
    le pipeline original — index JSON {arcname: [offset, size]} compris).
    Sortie :
        data/archives/frames_{split}_aug.tar(+.index.json)
        data/archives/flow_{split}_aug.tar(+.index.json)
    """
    archives_root.mkdir(parents=True, exist_ok=True)

    targets = [
        ("frames", out_frames_root / split, archives_root / f"frames_{split}_aug.tar"),
        ("flow",   out_flow_root   / split, archives_root / f"flow_{split}_aug.tar"),
    ]

    for modality, src_root, tar_path in targets:
        if not src_root.exists():
            print(f"  [skip-pack] {modality}/{split} : {src_root} introuvable")
            continue
        print(f"  -> {tar_path}")
        index = pack_split(src_root, tar_path)
        index_path = tar_path.with_suffix(tar_path.suffix + ".index.json")
        with open(index_path, "w") as f:
            json.dump(index, f)
        size_gb = tar_path.stat().st_size / 1e9
        print(f"     {len(index):,} fichiers | tar={size_gb:.2f} GB | "
              f"index={index_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames-root", default=None,
                        help=f"Défaut: {PROJECT_ROOT / 'data/frames'}")
    parser.add_argument("--flow-root", default=None,
                        help=f"Défaut: {PROJECT_ROOT / 'data/optical_flow'}")
    parser.add_argument("--out-frames", default=None,
                        help=f"Défaut: {PROJECT_ROOT / 'data/frames_augmented'}")
    parser.add_argument("--out-flow", default=None,
                        help=f"Défaut: {PROJECT_ROOT / 'data/optical_flow_augmented'}")
    parser.add_argument("--archives", default=None,
                        help=f"Destination des tars augmentés. Défaut: {PROJECT_ROOT / 'data/archives'}")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=list(SPLITS),
                        help="Splits à augmenter (défaut: train uniquement)")
    parser.add_argument("--copies", type=int, default=1,
                        help="Nombre de versions augmentées par vidéo (défaut: 1)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Taille de sortie carrée. Défaut 224 = taille des originaux, "
                             "à laisser tel quel pour rester compatible avec le pipeline.")
    parser.add_argument("--scale", type=float, nargs=2, default=(0.7, 1.0),
                        help="Range de scale du RandomResizedCrop")
    parser.add_argument("--ratio", type=float, nargs=2, default=(0.8, 1.25),
                        help="Range d'aspect ratio du RandomResizedCrop")
    parser.add_argument("--rotation-deg", type=float, default=10.0,
                        help="Amplitude max de rotation (degrés)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed pour reproduire le tirage des augmentations")
    parser.add_argument("--no-pack", action="store_true",
                        help="Saute l'étape d'empaquetage tar à la fin")
    args = parser.parse_args()

    # Résolution des chemins : si l'utilisateur passe un chemin relatif, on le
    # garde tel quel (relatif au CWD courant, comportement classique). Sinon on
    # utilise les valeurs par défaut ancrées sur PROJECT_ROOT pour que le script
    # marche indépendamment du dossier d'invocation.
    frames_root = Path(args.frames_root) if args.frames_root else PROJECT_ROOT / "data/frames"
    flow_root = Path(args.flow_root) if args.flow_root else PROJECT_ROOT / "data/optical_flow"
    out_frames_root = Path(args.out_frames) if args.out_frames else PROJECT_ROOT / "data/frames_augmented"
    out_flow_root = Path(args.out_flow) if args.out_flow else PROJECT_ROOT / "data/optical_flow_augmented"
    archives_root = Path(args.archives) if args.archives else PROJECT_ROOT / "data/archives"

    if not frames_root.is_dir():
        print(f"ERREUR : {frames_root} introuvable", file=sys.stderr)
        return 2
    if not flow_root.is_dir():
        print(f"ERREUR : {flow_root} introuvable", file=sys.stderr)
        return 2

    print("== Augmentation hors-ligne ==")
    print(f"  Source frames : {frames_root}")
    print(f"  Source flow   : {flow_root}")
    print(f"  Sortie frames : {out_frames_root}")
    print(f"  Sortie flow   : {out_flow_root}")
    print(f"  Splits        : {args.splits}")
    print(f"  Copies/vidéo  : {args.copies}")
    print(f"  image_size    : {args.image_size}")
    print(f"  scale         : {tuple(args.scale)}")
    print(f"  ratio         : {tuple(args.ratio)}")
    print(f"  rotation_deg  : {args.rotation_deg}")
    print(f"  workers       : {args.workers}")
    print(f"  seed          : {args.seed}")
    print()

    total_videos = 0
    all_errors: list[str] = []
    for split in args.splits:
        n, errs = augment_split(
            split=split,
            frames_root=frames_root,
            flow_root=flow_root,
            out_frames_root=out_frames_root,
            out_flow_root=out_flow_root,
            copies=args.copies,
            image_size=args.image_size,
            scale=tuple(args.scale),
            ratio=tuple(args.ratio),
            rotation_deg=args.rotation_deg,
            workers=args.workers,
            seed=args.seed,
        )
        total_videos += n
        all_errors.extend(errs)

    print()
    print(f"Génération terminée : {total_videos} vidéo(s) traitée(s), "
          f"{len(all_errors)} erreur(s)")
    if all_errors:
        for e in all_errors[:20]:
            print(f"  [err] {e}")
        if len(all_errors) > 20:
            print(f"  ... et {len(all_errors) - 20} autre(s)")

    if args.no_pack:
        print("\n--no-pack : empaquetage tar sauté.")
        return 0 if not all_errors else 1

    print("\n-- Empaquetage tar --")
    for split in args.splits:
        pack_augmented(split, out_frames_root, out_flow_root, archives_root)

    print("\nTerminé.")
    return 0 if not all_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
