"""
Empaquette frames/ et optical_flow/ en archives tar non compressées + index JSON
pour faciliter le transfert et permettre la lecture aléatoire sans dépaquetage.

Sortie (par défaut sous archives/) :
    frames_train.tar + frames_train.tar.index.json
    frames_val.tar   + frames_val.tar.index.json
    frames_test.tar  + frames_test.tar.index.json
    flow_train.tar   + flow_train.tar.index.json
    flow_val.tar     + flow_val.tar.index.json
    flow_test.tar    + flow_test.tar.index.json

L'index est un dict {nom_membre: [offset_data, size]} permettant un seek+read
direct dans le tar sans passer par tarfile.extract*().

Les JPG étant déjà compressés, le tar est laissé non compressé : taille
quasi identique à l'arborescence d'origine, et accès aléatoire conservé.

Usage :
    python pack_dataset.py
    python pack_dataset.py --frames-root frames --flow-root optical_flow --out archives
"""

import argparse
import json
import tarfile
from pathlib import Path

from tqdm import tqdm


def pack_split(src_root: Path, out_tar: Path) -> dict[str, list[int]]:
    """
    Empaquette tous les fichiers sous src_root dans out_tar (tar non compressé).
    Les arcnames sont relatifs à src_root.
    Retourne l'index {arcname: [offset_data, size]}.
    """
    files = sorted(p for p in src_root.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"Aucun fichier sous {src_root}")

    out_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar, "w") as tar:
        for f in tqdm(files, desc=out_tar.name, unit="file"):
            tar.add(f, arcname=str(f.relative_to(src_root)), recursive=False)

    index: dict[str, list[int]] = {}
    with tarfile.open(out_tar, "r") as tar:
        for m in tar.getmembers():
            if m.isfile():
                index[m.name] = [m.offset_data, m.size]
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", default="frames")
    parser.add_argument("--flow-root", default="optical_flow")
    parser.add_argument("--out", default="archives")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    flow_root = Path(args.flow_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = []
    for split in args.splits:
        targets.append(("frames", frames_root / split, out / f"frames_{split}.tar"))
        targets.append(("flow", flow_root / split, out / f"flow_{split}.tar"))

    for modality, src_root, tar_path in targets:
        if not src_root.exists():
            print(f"SKIP {modality}/{src_root.name} — {src_root} introuvable")
            continue
        print(f"-> {tar_path}")
        index = pack_split(src_root, tar_path)
        index_path = tar_path.with_suffix(tar_path.suffix + ".index.json")
        with open(index_path, "w") as f:
            json.dump(index, f)
        size_gb = tar_path.stat().st_size / 1e9
        print(f"   {len(index):,} fichiers | tar={size_gb:.2f} GB | index={index_path.name}")


if __name__ == "__main__":
    main()
