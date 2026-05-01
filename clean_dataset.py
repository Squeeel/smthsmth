"""
Nettoie frames/ et optical_flow/ avant de re-packer les archives.

Deux problèmes connus :

  1. Pollution flow dans frames/  (~48 vidéos de test, jamais train/val)
     compute_optical_flow.py a écrit certains flow_*.jpg directement dans
     frames/<split>/<video>/. Ces fichiers existent aussi (bit-exact) dans
     optical_flow/<split>/<video>/ — ce sont de purs doublons.
       -> doublon strict (même MD5) : suppression
       -> sinon (rare)               : déplacement vers optical_flow/

  2. Vidéos nativement incomplètes (~11 vidéos sur tous splits)
     Vidéos source trop courtes ou cassées : il manque des frames et/ou
     des paires de flow. Inutilisables pour le pipeline.
       -> suppression du dossier vidéo dans frames/ ET optical_flow/

Usage :
    uv run clean_dataset.py            # dry-run : affiche ce qui serait fait
    uv run clean_dataset.py --apply    # applique réellement
    uv run pack_dataset.py             # re-pack après nettoyage
"""

import argparse
import hashlib
import shutil
from pathlib import Path


EXPECTED_FRAMES = {f"frame_{i:03d}.jpg" for i in range(4)}
EXPECTED_FLOW = {f"flow_{c}_{i:03d}.jpg" for i in range(3) for c in ("x", "y")}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def list_video_dirs(frames_split: Path, split: str) -> list[Path]:
    """Liste tous les dossiers vidéo sous frames/<split>/."""
    if split == "test":
        return sorted(p for p in frames_split.iterdir() if p.is_dir())
    return sorted(
        v
        for c in frames_split.iterdir() if c.is_dir()
        for v in c.iterdir() if v.is_dir()
    )


def flow_dir_for(video: Path, frames_root: Path, flow_root: Path) -> Path:
    """Chemin attendu du dossier flow correspondant à un dossier vidéo."""
    return flow_root / video.relative_to(frames_root)


def clean(frames_root: Path, flow_root: Path, apply: bool) -> None:
    counts = {
        "flow_dup_removed": 0,    # flow dans frames/ identique à optical_flow/ -> rm
        "flow_moved": 0,          # flow dans frames/ pas dans optical_flow/    -> mv
        "videos_removed": 0,      # dossier vidéo incomplet -> rmtree
    }

    for split in ("train", "val", "test"):
        fr_split = frames_root / split
        if not fr_split.exists():
            continue
        print(f"\n=== {split} ===")

        # Phase 1 : flow files mal placés dans frames/<split>/<...>/<video>/
        for v in list_video_dirs(fr_split, split):
            misplaced = sorted(
                p for p in v.iterdir() if p.is_file() and p.name in EXPECTED_FLOW
            )
            if not misplaced:
                continue
            target_dir = flow_dir_for(v, frames_root, flow_root)
            for m in misplaced:
                target = target_dir / m.name
                if target.exists() and md5(m) == md5(target):
                    print(f"  rm DUP   {m}")
                    if apply:
                        m.unlink()
                    counts["flow_dup_removed"] += 1
                else:
                    print(f"  mv       {m} -> {target}")
                    if apply:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(m), str(target))
                    counts["flow_moved"] += 1

        # Phase 2 : vidéos incomplètes (ré-évalue après nettoyage du flow)
        for v in list_video_dirs(fr_split, split):
            if not v.exists():
                continue
            frames_present = {p.name for p in v.iterdir() if p.is_file()}
            fl_v = flow_dir_for(v, frames_root, flow_root)
            flow_present = (
                {p.name for p in fl_v.iterdir() if p.is_file()}
                if fl_v.exists() else set()
            )

            missing_fr = EXPECTED_FRAMES - frames_present
            missing_fl = EXPECTED_FLOW - flow_present
            if not missing_fr and not missing_fl:
                continue

            print(
                f"  rmtree   {v}"
                + (f"  (frames manquantes: {sorted(missing_fr)})" if missing_fr else "")
                + (f"  (flow manquant: {sorted(missing_fl)})" if missing_fl else "")
            )
            if apply:
                shutil.rmtree(v)
                if fl_v.exists():
                    shutil.rmtree(fl_v)
            counts["videos_removed"] += 1

    print("\n=== Bilan ===")
    print(f"  flow doublons supprimés  : {counts['flow_dup_removed']}")
    print(f"  flow déplacés            : {counts['flow_moved']}")
    print(f"  vidéos incomplètes rm    : {counts['videos_removed']}")
    if not apply:
        print("\n(dry-run — relance avec --apply pour effectuer le nettoyage)")
    else:
        print("\nMaintenant : re-packe avec  uv run pack_dataset.py")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-root", default="frames")
    p.add_argument("--flow-root", default="optical_flow")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    clean(Path(args.frames_root), Path(args.flow_root), apply=args.apply)


if __name__ == "__main__":
    main()
