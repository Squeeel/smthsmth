"""
Récap synthétique d'un run d'entraînement.

Lit le dossier d'un run (runs/<model>_<timestamp>/) et :
  - imprime hyperparams, métriques best/final, top/flop classes en console
  - sauve trois figures dans le dossier du run :
      report_curves.png       loss + acc + lr par epoch
      report_per_class.png    F1 par classe, trié décroissant
      report_confusion.png    matrice de confusion normalisée par ligne

Les noms de classes sont récupérés depuis frames/train/<NNN_NomClasse>/ si
disponible, sinon fallback en index numériques.

Usage :
    uv run report_run.py runs/CNN_two_stream_20260501-143022
    uv run report_run.py runs/<dir> --frames-root /chemin/vers/frames
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend non-interactif (marche sur serveur sans display)
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------ Chargement ------------------------------


def _build_names(pairs: list[tuple[int, str]]) -> list[str] | None:
    if not pairs:
        return None
    pairs = sorted(set(pairs))
    max_idx = pairs[-1][0]
    out = [f"class_{i:02d}" for i in range(max_idx + 1)]
    for i, name in pairs:
        out[i] = name
    return out


def class_names_from_archives(archives_root: Path) -> list[str] | None:
    """
    Construit la liste des noms de classes depuis l'index du tar train :
    chaque entrée est `<NNN_NomClasse>/<video>/<file>`.
    Retourne None si l'index n'existe pas ou n'est pas exploitable.
    """
    index_path = archives_root / "frames_train.tar.index.json"
    if not index_path.exists():
        return None
    with open(index_path) as f:
        index = json.load(f)
    pairs: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name in index:
        cls, sep, _ = name.partition("/")
        if not sep or cls in seen:
            continue
        seen.add(cls)
        idx_str, _, label = cls.partition("_")
        if idx_str.isdigit() and label:
            pairs.append((int(idx_str), label))
    return _build_names(pairs)


def class_names_from_frames(frames_root: Path) -> list[str] | None:
    """Idem, mais depuis frames/train/<NNN_NomClasse>/ sur le filesystem."""
    train_dir = frames_root / "train"
    if not train_dir.exists():
        return None
    pairs: list[tuple[int, str]] = []
    for d in train_dir.iterdir():
        if not d.is_dir():
            continue
        idx_str, _, name = d.name.partition("_")
        if idx_str.isdigit() and name:
            pairs.append((int(idx_str), name))
    return _build_names(pairs)


def get_class_names(
    frames_root: Path | None = None,
    archives_root: Path | None = None,
) -> list[str] | None:
    """
    Résout les noms de classes en essayant d'abord les archives (source de
    vérité au runtime), puis le filesystem `frames/`. Retourne None si rien
    n'est exploitable — l'appelant tombera sur les indices numériques.
    """
    if archives_root is not None:
        names = class_names_from_archives(archives_root)
        if names is not None:
            return names
    if frames_root is not None:
        return class_names_from_frames(frames_root)
    return None


def load_run(run_dir: Path) -> tuple[dict, list[dict], dict]:
    config = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    history: list[dict] = []
    history_path = run_dir / "history.csv"
    if history_path.exists():
        with open(history_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append({
                    k: int(v) if k == "epoch" else float(v)
                    for k, v in row.items()
                })
    return config, history, summary


# ------------------------------ Console ------------------------------


def short_class_name(idx: int, names: list[str] | None) -> str:
    if names is not None and idx < len(names):
        return names[idx]
    return f"class_{idx:02d}"


def print_recap(run_dir: Path, config: dict, history: list[dict], summary: dict,
                class_names: list[str] | None) -> None:
    args = config["args"]
    sysinfo = config["system"]
    model = config["model"]
    ds = config["dataset"]

    print(f"\n=== {run_dir.name} ===")
    print(f"Modèle      : {model['name']}  "
          f"(INPUT_MODE={model['input_mode']}, params={model['trainable_params']:,})")
    print(f"Démarré     : {config['started_at']}")
    if "finished_at" in summary:
        print(f"Terminé     : {summary['finished_at']}")
    print(f"Durée       : {summary.get('total_time_s', 0):.0f}s sur "
          f"{summary.get('n_epochs_run', '?')} epochs"
          + ("  (early stopped)" if summary.get("early_stopped") else ""))

    print("\nHyperparams :")
    for k in ("lr", "batch_size", "weight_decay", "image_size",
              "epochs", "patience", "num_workers"):
        if k in args:
            print(f"  {k:14s} = {args[k]}")
    print(f"Dataset     : train={ds['train_size']:,}  val={ds['val_size']:,}  "
          f"num_classes={ds.get('num_classes', '?')}")

    gpu = sysinfo.get("gpu_name", "CPU")
    print(f"\nSystème     : {gpu}  |  torch {sysinfo.get('torch', '?')}"
          + (f"  |  cuda {sysinfo['cuda']}" if "cuda" in sysinfo else ""))
    if "git_commit" in sysinfo:
        dirty = " (dirty)" if sysinfo.get("git_dirty") else ""
        print(f"Git         : {sysinfo['git_commit'][:8]}{dirty}")

    print("\nMétriques :")
    print(f"  best val top1   : {summary.get('best_val_top1_acc', 0):.4f}  "
          f"(epoch {summary.get('best_epoch', '?')})")
    if "final_val_top1_acc" in summary:
        print(f"  final val top1  : {summary['final_val_top1_acc']:.4f}")
        print(f"  final val top5  : {summary['final_val_top5_acc']:.4f}")
    if history:
        last = history[-1]
        gap = last["train_acc"] - last["val_top1_acc"]
        print(f"  dernière epoch  : train={last['train_acc']:.4f}  "
              f"val={last['val_top1_acc']:.4f}  (gap={gap:+.4f})")

    if "per_class" in summary:
        per_class = summary["per_class"]
        valid = [c for c in per_class if c["support"] > 0]
        sorted_pc = sorted(valid, key=lambda c: c["f1"], reverse=True)

        def fmt(c):
            name = short_class_name(c["class"], class_names)
            return (f"  {c['class']:3d}  {name[:42]:42s}  "
                    f"F1={c['f1']:.3f}  P={c['precision']:.3f}  "
                    f"R={c['recall']:.3f}  n={c['support']}")

        n_show = min(5, len(sorted_pc))
        print(f"\nTop {n_show} classes (F1) :")
        for c in sorted_pc[:n_show]:
            print(fmt(c))
        print(f"\nFlop {n_show} classes (F1) :")
        for c in sorted_pc[-n_show:]:
            print(fmt(c))


# ------------------------------ Figures ------------------------------


def plot_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train", linewidth=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val", linewidth=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train top1", linewidth=2)
    axes[1].plot(epochs, [h["val_top1_acc"] for h in history], label="val top1", linewidth=2)
    axes[1].plot(epochs, [h["val_top5_acc"] for h in history], label="val top5",
                 linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 1)

    axes[2].plot(epochs, [h["lr"] for h in history], color="tab:purple")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("learning rate")
    axes[2].set_yscale("log")
    axes[2].set_title("LR schedule")
    axes[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_per_class(per_class: list[dict], out_path: Path,
                   class_names: list[str] | None) -> None:
    valid = [c for c in per_class if c["support"] > 0]
    classes = sorted(valid, key=lambda c: c["f1"], reverse=True)
    n = len(classes)
    f1 = [c["f1"] for c in classes]
    supports = [c["support"] for c in classes]
    labels = [
        f"{c['class']:02d} {short_class_name(c['class'], class_names)[:32]}  (n={c['support']})"
        for c in classes
    ]

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.22)))
    bars = ax.barh(range(n), f1, color="steelblue")
    # Coloration : rouge pour les pires, vert pour les meilleurs
    for bar, val in zip(bars, f1):
        if val < 0.1:
            bar.set_color("indianred")
        elif val > 0.6:
            bar.set_color("seagreen")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1")
    ax.set_xlim(0, 1)
    ax.axvline(np.mean(f1), color="black", linestyle="--", linewidth=1,
               label=f"moyenne F1 = {np.mean(f1):.3f}")
    ax.set_title(f"F1 par classe ({n} classes)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion(cm: list[list[int]], out_path: Path,
                   class_names: list[str] | None,
                   title: str | None = None) -> None:
    cm_arr = np.asarray(cm, dtype=float)
    n = cm_arr.shape[0]
    # On masque les classes sans aucun support (pas de ground-truth) — sinon
    # leur ligne entière apparaît à zéro et les couleurs deviennent illisibles.
    has_support = cm_arr.sum(axis=1) > 0
    valid_idx = np.where(has_support)[0]
    cm_valid = cm_arr[valid_idx][:, valid_idx]
    row_sums = cm_valid.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_valid / row_sums

    n_v = len(valid_idx)
    fig, ax = plt.subplots(figsize=(min(14, n_v * 0.4 + 2),
                                     min(14, n_v * 0.4 + 2)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, label="recall")

    labels = [
        f"{i:02d} " + short_class_name(i, class_names)[:24]
        for i in valid_idx
    ]
    ax.set_xticks(range(n_v))
    ax.set_yticks(range(n_v))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    if title is None:
        title = f"Matrice de confusion (normalisée par ligne, {n_v} classes)"
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def save_confusion_csv(cm: list[list[int]], out_path: Path,
                       class_names: list[str] | None) -> None:
    """
    Sauve la matrice de confusion brute (counts entiers, non normalisés) en CSV.

    Format :
        - 1ère ligne : en-tête, ""  puis  N noms de classes prédites
        - lignes suivantes : nom de classe vraie + N counts

    Pas de masquage des classes à support nul : on conserve la matrice complète
    pour que tout post-traitement (pandas, Excel) parte de la vérité brute.
    """
    cm_arr = np.asarray(cm, dtype=int)
    n = cm_arr.shape[0]
    labels = [
        f"{i:02d} {short_class_name(i, class_names)}"
        for i in range(n)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true \\ pred", *labels])
        for i in range(n):
            w.writerow([labels[i], *cm_arr[i].tolist()])


# ------------------------------ Entrée publique ------------------------------


def generate_report(run_dir: Path, frames_root: Path = Path("data/frames"),
                    archives_root: Path = Path("data/archives"),
                    verbose: bool = True) -> None:
    """
    Charge un run et produit son rapport : récap console + 3 figures PNG
    sauvegardées dans le dossier du run.

    Cette fonction est appelée automatiquement à la fin d'un entraînement
    (cf. main.py) et utilisable en standalone via le CLI ci-dessous.
    """
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"pas un dossier de run valide : {run_dir}")

    config, history, summary = load_run(run_dir)
    class_names = get_class_names(frames_root=frames_root, archives_root=archives_root)

    if verbose:
        print_recap(run_dir, config, history, summary, class_names)

    figs: list[Path] = []
    if history:
        out = run_dir / "report_curves.png"
        plot_curves(history, out)
        figs.append(out)
    if "per_class" in summary:
        out = run_dir / "report_per_class.png"
        plot_per_class(summary["per_class"], out, class_names)
        figs.append(out)
    if "confusion_matrix" in summary:
        out = run_dir / "report_confusion.png"
        plot_confusion(summary["confusion_matrix"], out, class_names)
        figs.append(out)
        out_csv = run_dir / "report_confusion.csv"
        save_confusion_csv(summary["confusion_matrix"], out_csv, class_names)
        figs.append(out_csv)

    if verbose and figs:
        print("\nFigures :")
        for f in figs:
            print(f"  {f}")


# ------------------------------ CLI ------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Dossier d'un run (runs/<...>/)")
    parser.add_argument("--frames-root", default="data/frames",
                        help="Pour récupérer les noms de classes (défaut: data/frames).")
    parser.add_argument("--archives", default="data/archives",
                        help="Source alternative des noms de classes via l'index tar (défaut: data/archives).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"introuvable : {run_dir}")
    generate_report(run_dir, frames_root=Path(args.frames_root),
                    archives_root=Path(args.archives))


if __name__ == "__main__":
    main()
