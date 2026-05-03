"""
Intègre les nouvelles données de `frame_new/` dans le dataset existant
(`frames/` + `optical_flow/`) en réutilisant tous les calculs déjà faits.

Constat (vérifié) :
  - `frame_new/` reproduit la même arborescence que `frames/` (32 classes en
    train/val, vidéos à plat en test).
  - Les vidéos communes sont bit-exact identiques entre les deux roots → leur
    flux optique déjà calculé reste valide.
  - Seules quelques vidéos sont *nouvelles* (présentes dans `frame_new/` et
    absentes de `frames/`) ; pour celles-ci, et pour celles-ci seulement, il
    faut copier les frames puis calculer le flux TVL1.

Stratégie (sûre par défaut) :
  1. Discovery : liste les vidéos nouvelles par split en comparant deux ensembles
     de chemins relatifs. Warn si une vidéo a *disparu* (ne supprime jamais).
  2. Sanity-check : sur les vidéos communes, vérifie que les 4 frames existent
     des deux côtés ; option `--md5` pour comparer les empreintes (lent).
  3. Copie : pour chaque vidéo nouvelle, copie le dossier
     `frame_new/{split}/<...>/<video>/` vers `frames/{split}/<...>/<video>/`.
     N'écrase JAMAIS un dossier existant.
  4. Flux : calcule le flux uniquement pour les vidéos nouvelles, en
     réutilisant `process_video_folder` de `compute_optical_flow.py` (même
     normalisation, même clip à ±20, même format JPG).
  5. Dry-run par défaut. Utiliser `--apply` pour exécuter réellement.
  6. Ne touche pas aux archives `archives/` — relancer `pack_dataset.py` à la
     fin (rappel imprimé).

Usage :
    uv run update_dataset.py                 # dry-run
    uv run update_dataset.py --apply         # applique
    uv run update_dataset.py --apply --md5   # + vérif MD5 des vidéos communes
    uv run update_dataset.py --apply --workers 8

Après application :
    uv run pack_dataset.py                   # re-génère les archives tar
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

# Réutilise EXACTEMENT le pipeline de calcul existant (clip ±20, JPG, etc.).
from compute_optical_flow import process_video_folder


EXPECTED_FRAMES = {f"frame_{i:03d}.jpg" for i in range(4)}
EXPECTED_FLOW = {f"flow_{c}_{i:03d}.jpg" for i in range(3) for c in ("x", "y")}
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_video_relpaths(root: Path, split: str) -> set[Path]:
    """
    Renvoie les chemins relatifs (depuis `root`) de tous les dossiers vidéo
    sous `root/{split}`. Train/val sont organisés en classes ; test est à plat.
    """
    split_dir = root / split
    if not split_dir.is_dir():
        return set()

    if split == "test":
        return {v.relative_to(root) for v in split_dir.iterdir() if v.is_dir()}

    return {
        v.relative_to(root)
        for c in split_dir.iterdir() if c.is_dir()
        for v in c.iterdir() if v.is_dir()
    }


def diff_videos(frame_new_root: Path, frames_root: Path, split: str) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Retourne (added, removed, common) : chemins relatifs depuis le root parent.
    `added`   = dans frame_new mais pas dans frames -> à copier + flux à calculer
    `removed` = dans frames mais pas dans frame_new -> on ne touche PAS, juste warn
    `common`  = des deux côtés -> à sanity-checker
    """
    new_set = list_video_relpaths(frame_new_root, split)
    old_set = list_video_relpaths(frames_root, split)
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    common = sorted(new_set & old_set)
    return added, removed, common


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_video_complete(video_dir: Path) -> list[str]:
    """Retourne la liste des frames manquantes (vide = vidéo complète)."""
    if not video_dir.is_dir():
        return sorted(EXPECTED_FRAMES)
    present = {p.name for p in video_dir.glob("frame_*.jpg")}
    return sorted(EXPECTED_FRAMES - present)


def check_flow_complete(flow_dir: Path) -> list[str]:
    """Retourne la liste des flow manquants (vide = vidéo complète côté flux)."""
    if not flow_dir.is_dir():
        return sorted(EXPECTED_FLOW)
    present = {p.name for p in flow_dir.glob("flow_*.jpg")}
    return sorted(EXPECTED_FLOW - present)


def sanity_common(common: list[Path], frame_new_root: Path, frames_root: Path, do_md5: bool) -> list[str]:
    """
    Vérifie l'invariant : pour chaque vidéo commune, les 4 frames sont là des
    deux côtés. Avec `do_md5`, vérifie aussi qu'elles sont bit-exact.
    Retourne la liste des problèmes détectés (vide = OK).
    """
    issues: list[str] = []
    for relpath in tqdm(common, desc="sanity-check (common)", unit="video", leave=False):
        new_dir = frame_new_root / relpath
        old_dir = frames_root / relpath

        miss_new = check_video_complete(new_dir)
        miss_old = check_video_complete(old_dir)
        if miss_new:
            issues.append(f"{relpath}: frames manquantes côté frame_new : {miss_new}")
        if miss_old:
            issues.append(f"{relpath}: frames manquantes côté frames : {miss_old}")

        if do_md5 and not miss_new and not miss_old:
            for name in sorted(EXPECTED_FRAMES):
                if md5_file(new_dir / name) != md5_file(old_dir / name):
                    issues.append(f"{relpath}/{name}: MD5 diffère entre frame_new et frames")
    return issues


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------

def copy_new_videos(
    added_by_split: dict[str, list[Path]],
    frame_new_root: Path,
    frames_root: Path,
    apply: bool,
) -> tuple[list[Path], list[tuple[Path, list[str]]]]:
    """
    Copie chaque dossier vidéo nouveau de `frame_new/<...>` vers `frames/<...>`.
    Refuse d'écraser un dossier existant. Crée les dossiers parents au besoin.

    Retourne :
      - copied   : relpaths effectivement copiés (ou copiables en dry-run).
      - skipped  : (relpath, raison) — vidéos sources incomplètes ou destination déjà
                   présente. Ces relpaths ne doivent PAS alimenter le calcul de flux.
    """
    copied: list[Path] = []
    skipped: list[tuple[Path, list[str]]] = []
    for added in added_by_split.values():
        for relpath in added:
            src = frame_new_root / relpath
            dst = frames_root / relpath

            # Sanity côté source : on ne touche que des vidéos avec 4 frames.
            miss = check_video_complete(src)
            if miss:
                print(f"  [skip] {relpath} : source incomplète, frames manquantes : {miss}")
                skipped.append((relpath, miss))
                continue

            if dst.exists():
                # Ne devrait jamais arriver (sinon ce serait un common, pas un added).
                print(f"  [skip] {relpath} : destination déjà présente — non écrasée")
                skipped.append((relpath, ["destination existe"]))
                continue

            copied.append(relpath)
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dst)
            print(f"  [copy{' ' if apply else ' DRY'}] {src} -> {dst}")
    return copied, skipped


# ---------------------------------------------------------------------------
# Optical flow (réutilise compute_optical_flow.process_video_folder)
# ---------------------------------------------------------------------------

def build_flow_tasks(copied: list[Path], frames_root: Path, flow_root: Path) -> list[tuple[Path, Path]]:
    """
    Pour chaque vidéo *effectivement copiée*, construit la paire
    (video_dir, output_flow_dir) attendue par `process_video_folder`. Le
    video_dir est lu depuis frames/ (post-copie) afin de garder le pipeline
    strictement identique à celui appliqué au reste du dataset.
    """
    return [(frames_root / r, flow_root / r) for r in copied]


def compute_flows(tasks: list[tuple[Path, Path]], workers: int, apply: bool) -> tuple[int, list[str]]:
    """
    Calcule le flux optique pour la liste de tâches données. En dry-run, on
    n'exécute pas mais on liste ce qui serait fait. On refuse d'écraser un flow
    déjà partiellement calculé (sécurité paranoid).
    """
    errors: list[str] = []
    to_run: list[tuple[Path, Path]] = []
    for video_dir, output_dir in tasks:
        if output_dir.exists() and any(output_dir.glob("flow_*.jpg")):
            # Si tout est déjà là (6 fichiers attendus), on saute simplement.
            missing = check_flow_complete(output_dir)
            if not missing:
                print(f"  [skip] flow déjà complet : {output_dir}")
                continue
            errors.append(
                f"flow partiel déjà présent : {output_dir} (manque {missing}). "
                "Vérifie manuellement avant de relancer."
            )
            continue
        to_run.append((video_dir, output_dir))
        print(f"  [flow{' ' if apply else ' DRY'}] {video_dir} -> {output_dir}")

    if not apply or not to_run:
        return len(to_run), errors

    # Réutilise rigoureusement le worker du script principal.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for result in tqdm(ex.map(process_video_folder, to_run), total=len(to_run), unit="video", desc="optical flow"):
            if result is not None:
                errors.append(result)

    return len(to_run), errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame-new-root", default="data/frame_new", help="Source des nouvelles données (défaut: data/frame_new)")
    parser.add_argument("--frames-root", default="data/frames", help="Destination des frames consolidées (défaut: data/frames)")
    parser.add_argument("--flow-root", default="data/optical_flow", help="Destination du flux (défaut: data/optical_flow)")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    parser.add_argument("--workers", type=int, default=4, help="Workers pour le calcul du flux (défaut: 4)")
    parser.add_argument("--apply", action="store_true", help="Applique réellement les modifications (sinon dry-run)")
    parser.add_argument("--md5", action="store_true", help="Vérifie MD5 des vidéos communes (lent — désactivé par défaut)")
    parser.add_argument("--skip-sanity", action="store_true", help="Saute le sanity-check des vidéos communes")
    args = parser.parse_args()

    frame_new_root = Path(args.frame_new_root)
    frames_root = Path(args.frames_root)
    flow_root = Path(args.flow_root)

    if not frame_new_root.is_dir():
        print(f"ERREUR : {frame_new_root} introuvable", file=sys.stderr)
        return 2
    if not frames_root.is_dir():
        print(f"ERREUR : {frames_root} introuvable", file=sys.stderr)
        return 2

    print(f"== Mode : {'APPLY' if args.apply else 'DRY-RUN'} ==")
    print(f"  frame_new : {frame_new_root}")
    print(f"  frames    : {frames_root}")
    print(f"  flow      : {flow_root}")
    print()

    # 1) Discovery.
    added_by_split: dict[str, list[Path]] = {}
    common_by_split: dict[str, list[Path]] = {}
    total_added = total_removed = 0
    print("-- Discovery --")
    for split in args.splits:
        added, removed, common = diff_videos(frame_new_root, frames_root, split)
        added_by_split[split] = added
        common_by_split[split] = common
        total_added += len(added)
        total_removed += len(removed)
        print(f"  {split:5s}: +{len(added):4d} nouveau(x), -{len(removed):4d} supprimé(s), {len(common):5d} commun(s)")
        for r in removed:
            print(f"    [warn] disparue dans frame_new (non touchée) : {r}")
        for a in added:
            print(f"    [add ] {a}")
    print()

    if total_added == 0:
        print("Rien à faire — aucune vidéo nouvelle détectée.")
        return 0

    # 2) Sanity-check.
    if not args.skip_sanity:
        print("-- Sanity-check vidéos communes --")
        for split in args.splits:
            issues = sanity_common(common_by_split[split], frame_new_root, frames_root, args.md5)
            if issues:
                print(f"  {split}: {len(issues)} problème(s) :")
                for i in issues[:20]:
                    print(f"    {i}")
                if len(issues) > 20:
                    print(f"    ... et {len(issues) - 20} autre(s)")
                print(f"  STOP — corrige ces problèmes avant de relancer (ou utilise --skip-sanity).")
                return 1
        print("  OK")
        print()

    # 3) Copie des nouvelles vidéos.
    print("-- Copie des nouvelles vidéos (frame_new -> frames) --")
    copied, skipped = copy_new_videos(added_by_split, frame_new_root, frames_root, args.apply)
    print(f"  {len(copied)} vidéo(s) {'copiée(s)' if args.apply else 'à copier'}, {len(skipped)} vidéo(s) écartée(s)")
    print()

    # 4) Flux optique uniquement pour les vidéos *effectivement* copiées.
    print("-- Flux optique TVL1 (vidéos nouvelles uniquement) --")
    tasks = build_flow_tasks(copied, frames_root, flow_root)
    n_flow, errors = compute_flows(tasks, args.workers, args.apply)
    print(f"  {n_flow} vidéo(s) {'traitée(s)' if args.apply else 'à traiter'}")
    if errors:
        print(f"  {len(errors)} erreur(s)/avertissement(s) :")
        for e in errors:
            print(f"    {e}")
    print()

    # 5) Rappel.
    if args.apply:
        print("Terminé. Étape suivante :")
        print("  uv run pack_dataset.py    # re-génère archives/ — main.py lit depuis là")
    else:
        print("Dry-run terminé. Relance avec --apply pour appliquer.")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
