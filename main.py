"""
Pipeline d'entraînement généraliste pour smthsmth.

Pas de registre dans main : un modèle = un fichier dans models_code/ qui expose
deux symboles standard.

Convention de chaque fichier `models_code/<NomDuModele>.py` :
    INPUT_MODE : str
        Mode de chargement attendu par le modèle. Doit être l'un de :
            "rgb_first" : 1ère frame RGB                              -> (B, 3, H, W)
            "rgb_stack" : 4 frames RGB empilées                        -> (B, 12, H, W)
            "flow"      : 3 paires (flow_x, flow_y) empilées           -> (B, 6, H, W)
            "two_stream": (frames RGB, flow)                           -> (B, 4, 3, H, W) + (B, 3, 2, H, W)
    build(num_classes: int, **kwargs) -> nn.Module
        Factory qui retourne le modèle prêt à être entraîné.

Usage :
    python pack_dataset.py                              # une seule fois : crée archives/
    python main.py --model CNN_rgb
    python main.py --model CNN_flow
    python main.py --model CNN_two_stream --archives archives
    (n'importe quel fichier de models_code/ qui respecte la convention)

Les données sont lues depuis archives/{frames,flow}_{split}.tar via un index JSON
(pas d'extraction). Voir pack_dataset.py pour générer ces archives.
"""

import argparse
import csv
import importlib
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# ------------------------------ Lecture random-access dans un tar ------------------------------

class TarReader:
    """
    Lit des membres d'un tar non compressé via un index pré-calculé
    {nom_membre: [offset_data, size]}, sans dépaqueter.

    Le file descriptor est ouvert paresseusement à la première lecture pour rester
    fork-safe avec le DataLoader multi-workers : tant que `__init__` n'ouvre pas
    le fd, chaque worker ouvre le sien après le fork (pas de seek partagé).
    """

    def __init__(self, tar_path: str | Path, index_path: str | Path | None = None):
        self.tar_path = str(tar_path)
        if index_path is None:
            index_path = self.tar_path + ".index.json"
        with open(index_path) as f:
            self.index: dict[str, list[int]] = json.load(f)
        self._fd = None

    def names(self):
        return self.index.keys()

    def read(self, name: str) -> bytes:
        if self._fd is None:
            self._fd = open(self.tar_path, "rb")
        offset, size = self.index[name]
        self._fd.seek(offset)
        return self._fd.read(size)

    def __getstate__(self):
        # Ne pas pickler le fd (cas "spawn") — il sera réouvert dans le worker.
        state = self.__dict__.copy()
        state["_fd"] = None
        return state


# ------------------------------ Dataset ------------------------------

class SmthSmthDataset(Dataset):
    """
    Lit les vidéos depuis les archives tar produites par pack_dataset.py.

    Retourne TOUJOURS ((tenseurs...), label).
        single-stream : ((x,), label)              -> model(*inputs) == model(x)
        two-stream    : ((frames, flow), label)    -> model(*inputs) == model(frames, flow)

    Pour le split "test", `label = -1` (pas d'annotation).
    """

    VALID_MODES = {"rgb_first", "rgb_stack", "flow", "two_stream"}

    def __init__(
        self,
        archives_root: str = "archives",
        split: str = "train",
        mode: str = "rgb_first",
        image_size: int = 112,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode invalide: {mode} (valides: {self.VALID_MODES})")
        self.mode = mode
        self.split = split
        self.image_size = image_size

        archives_root = Path(archives_root)
        self.frames_reader = TarReader(archives_root / f"frames_{split}.tar")
        self.flow_reader: TarReader | None = (
            TarReader(archives_root / f"flow_{split}.tar")
            if mode in {"flow", "two_stream"}
            else None
        )

        # Inventaire des vidéos depuis l'index :
        #   train/val : "<class>/<video>/<file>"
        #   test      : "<video>/<file>"
        # Les archives sont supposées propres (cf. clean_dataset.py) : toute
        # vidéo présente a ses 4 frames + 6 flow files.
        videos: dict[str, int] = {}
        for name in self.frames_reader.names():
            parts = name.split("/")
            if split == "test":
                if len(parts) != 2:
                    continue
                videos.setdefault(parts[0], -1)
            else:
                if len(parts) != 3:
                    continue
                video_arc = f"{parts[0]}/{parts[1]}"
                if video_arc not in videos:
                    videos[video_arc] = int(parts[0].split("_")[0])

        # Liste triée stable : (arcname_video, class_idx)
        self.samples: list[tuple[str, int]] = sorted(videos.items())

        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        self.norm_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.norm_flow = transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, name: str) -> torch.Tensor:
        img = Image.open(io.BytesIO(self.frames_reader.read(name))).convert("RGB")
        return self.norm_rgb(self.to_tensor(self.resize(img)))

    def _load_gray(self, name: str) -> torch.Tensor:
        assert self.flow_reader is not None
        img = Image.open(io.BytesIO(self.flow_reader.read(name))).convert("L")
        return self.to_tensor(self.resize(img))

    def _load_rgb_stack(self, video_arc: str) -> torch.Tensor:
        # (4, 3, H, W)
        return torch.stack([self._load_rgb(f"{video_arc}/frame_{i:03d}.jpg") for i in range(4)])

    def _load_flow_stack(self, video_arc: str) -> torch.Tensor:
        # (3, 2, H, W)
        pairs = []
        for i in range(3):
            fx = self._load_gray(f"{video_arc}/flow_x_{i:03d}.jpg")
            fy = self._load_gray(f"{video_arc}/flow_y_{i:03d}.jpg")
            pairs.append(self.norm_flow(torch.cat([fx, fy], dim=0)))
        return torch.stack(pairs)

    def __getitem__(self, idx: int):
        video_arc, label = self.samples[idx]

        if self.mode == "rgb_first":
            return (self._load_rgb(f"{video_arc}/frame_000.jpg"),), label

        if self.mode == "rgb_stack":
            stacked = self._load_rgb_stack(video_arc).flatten(0, 1)  # (12, H, W)
            return (stacked,), label

        if self.mode == "flow":
            stacked = self._load_flow_stack(video_arc).flatten(0, 1)  # (6, H, W)
            return (stacked,), label

        # two_stream
        frames = self._load_rgb_stack(video_arc)   # (4, 3, H, W)
        flow = self._load_flow_stack(video_arc)    # (3, 2, H, W)
        return (frames, flow), label


# ------------------------------ Chargement dynamique du modèle ------------------------------

def load_model(name: str, num_classes: int) -> tuple[nn.Module, str]:
    """
    Charge `models_code/<name>.py` et appelle son `build(num_classes=...)`.
    Retourne (modèle, INPUT_MODE).
    """
    try:
        mod = importlib.import_module(f"models_code.{name}")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"models_code/{name}.py introuvable") from e

    if not hasattr(mod, "build"):
        raise AttributeError(f"models_code/{name}.py doit exposer build(num_classes, **kwargs)")
    if not hasattr(mod, "INPUT_MODE"):
        raise AttributeError(f"models_code/{name}.py doit exposer INPUT_MODE (str)")

    return mod.build(num_classes=num_classes), mod.INPUT_MODE


# ------------------------------ Train / eval ------------------------------

def _to_device(inputs, device):
    return tuple(t.to(device, non_blocking=True) for t in inputs)


# ------------------------------ Logging utilitaires ------------------------------

def make_run_dir(save_root: Path, model_name: str) -> Path:
    """runs/<model>_<YYYYMMDD-HHMMSS>/, suffixe -2/-3/... si collision."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_root.mkdir(parents=True, exist_ok=True)
    candidate = save_root / f"{model_name}_{stamp}"
    n = 2
    while candidate.exists():
        candidate = save_root / f"{model_name}_{stamp}-{n}"
        n += 1
    candidate.mkdir()
    return candidate


def collect_system_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip())
        info["git_commit"] = commit
        info["git_dirty"] = dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def topk_correct(logits: torch.Tensor, y: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return pred.eq(y.unsqueeze(1)).any(dim=1).sum().item()


@torch.no_grad()
def final_eval(model, loader, num_classes: int, device) -> dict:
    """Eval finale : top1, top5, matrice de confusion, métriques par classe."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    top1, top5, total = 0, 0, 0
    for inputs, y in tqdm(loader, desc="final-eval", leave=False):
        inputs = _to_device(inputs, device)
        y = y.to(device, non_blocking=True)
        logits = model(*inputs)
        preds = logits.argmax(1)
        for t, p in zip(y.cpu().tolist(), preds.cpu().tolist()):
            cm[t, p] += 1
        top1 += (preds == y).sum().item()
        top5 += topk_correct(logits, y, 5)
        total += y.size(0)

    cm_list = cm.tolist()
    per_class = []
    for c in range(num_classes):
        tp = cm_list[c][c]
        fp = sum(cm_list[r][c] for r in range(num_classes)) - tp
        fn = sum(cm_list[c]) - tp
        support = sum(cm_list[c])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class.append({
            "class": c,
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })

    return {
        "top1": top1 / total,
        "top5": top5 / total,
        "confusion_matrix": cm_list,
        "per_class": per_class,
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, y in tqdm(loader, desc="train", leave=False):
        inputs = _to_device(inputs, device)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(*inputs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss, top1, top5, total = 0.0, 0, 0, 0
    for inputs, y in tqdm(loader, desc="val", leave=False):
        inputs = _to_device(inputs, device)
        y = y.to(device, non_blocking=True)
        logits = model(*inputs)
        loss = criterion(logits, y)

        preds = logits.argmax(1)
        # Accumulation vectorisée de la matrice de confusion (sur CPU).
        flat_idx = (y * num_classes + preds).cpu()
        cm.view(-1).index_add_(
            0, flat_idx, torch.ones(flat_idx.numel(), dtype=torch.int64)
        )

        bs = y.size(0)
        total_loss += loss.item() * bs
        top1 += (preds == y).sum().item()
        top5 += topk_correct(logits, y, 5)
        total += bs
    return total_loss / total, top1 / total, top5 / total, cm.tolist()


# ------------------------------ Main ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="CNN_rgb",
        help="Nom du fichier (sans .py) dans models_code/. "
             "Doit exposer INPUT_MODE et build(num_classes).",
    )
    parser.add_argument("--archives", default="archives",
                        help="Dossier contenant frames_{split}.tar / flow_{split}.tar et leurs .index.json")
    parser.add_argument("--num-classes", type=int, default=33)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="runs",
                        help="Racine où créer le dossier du run (un sous-dossier par run, jamais écrasé).")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping : nb d'epochs sans amélioration de val_acc avant arrêt. 0 = désactivé.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Amélioration minimale de val_acc pour reset le compteur de patience.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, input_mode = load_model(args.model, args.num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} | INPUT_MODE={input_mode} | params={n_params:,}")

    train_set = SmthSmthDataset(args.archives, "train", input_mode, args.image_size)
    val_set = SmthSmthDataset(args.archives, "val", input_mode, args.image_size)
    print(f"Train: {len(train_set):,} samples | Val: {len(val_set):,} samples")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = make_run_dir(Path(args.save_dir), args.model)
    ckpt_path = run_dir / "best.pth"
    history_path = run_dir / "history.csv"
    cm_dir = run_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)
    best_cm_path = run_dir / "confusion_best.png"
    print(f"Run dir: {run_dir}")

    # Pour le rendu des matrices de confusion par epoch. Si matplotlib / les
    # noms de classes ne sont pas dispos, on tombe en silence sur les indices
    # numériques et on ne plante pas l'entraînement.
    try:
        from report_run import plot_confusion, get_class_names
        class_names = get_class_names(
            frames_root=Path("frames"), archives_root=Path(args.archives)
        )
    except Exception as e:
        print(f"(plot CM désactivé : {type(e).__name__}: {e})")
        plot_confusion = None
        class_names = None

    def save_confusion(cm, out_path, title):
        if plot_confusion is None:
            return
        try:
            plot_confusion(cm, out_path, class_names, title=title)
        except Exception as e:
            print(f"(plot CM échoué pour {out_path.name}: {type(e).__name__}: {e})")

    config = {
        "args": vars(args),
        "model": {"name": args.model, "input_mode": input_mode, "trainable_params": n_params},
        "dataset": {
            "train_size": len(train_set),
            "val_size": len(val_set),
            "image_size": args.image_size,
            "num_classes": args.num_classes,
        },
        "system": collect_system_info(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history_fields = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_top1_acc", "val_top5_acc",
        "lr", "epoch_time_s",
    ]
    with open(history_path, "w", newline="") as f:
        csv.writer(f).writerow(history_fields)

    t0 = time.time()
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_top1, val_top5, val_cm = evaluate(
            model, val_loader, criterion, device, args.num_classes
        )
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_time = time.time() - epoch_t0

        # Matrice de confusion de l'epoch : un PNG par epoch dans confusion_matrices/.
        save_confusion(
            val_cm,
            cm_dir / f"epoch_{epoch:03d}.png",
            title=f"Epoch {epoch} — val top1 {val_top1:.4f}",
        )

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_top1:.6f}", f"{val_top5:.6f}",
                f"{lr:.6e}", f"{epoch_time:.2f}",
            ])

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} top1 {val_top1:.4f} top5 {val_top5:.4f} | "
            f"lr {lr:.2e} | {epoch_time:.0f}s"
        )

        if val_top1 > best_acc + args.min_delta:
            best_acc = val_top1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "val_acc": val_top1,
                    "val_top5_acc": val_top5,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            # Matrice du meilleur epoch en racine du run, écrasée à chaque amélioration.
            save_confusion(
                val_cm,
                best_cm_path,
                title=f"Best (epoch {epoch}) — val top1 {val_top1:.4f}",
            )
            print(f"  -> checkpoint sauvegardé ({ckpt_path}, val acc {val_top1:.4f})")
        else:
            epochs_without_improvement += 1
            if args.patience > 0:
                print(f"  -> pas d'amélioration ({epochs_without_improvement}/{args.patience})")
                if epochs_without_improvement >= args.patience:
                    print(f"Early stopping à l'epoch {epoch} (patience={args.patience}).")
                    early_stopped = True
                    break

    total_time = time.time() - t0

    summary = {
        "best_val_top1_acc": best_acc,
        "best_epoch": best_epoch,
        "total_time_s": round(total_time, 1),
        "n_epochs_run": epoch,
        "early_stopped": early_stopped,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    if ckpt_path.exists():
        print("\nÉvaluation finale avec le best checkpoint…")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        final = final_eval(model, val_loader, args.num_classes, device)
        summary["final_val_top1_acc"] = final["top1"]
        summary["final_val_top5_acc"] = final["top5"]
        summary["per_class"] = final["per_class"]
        summary["confusion_matrix"] = final["confusion_matrix"]

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest val acc: {best_acc:.4f} (epoch {best_epoch}) | total {total_time:.0f}s")
    print(f"Run sauvegardé : {run_dir}")

    # Rapport automatique : récap console + figures (curves, per_class, confusion)
    try:
        from report_run import generate_report
        generate_report(run_dir, archives_root=Path(args.archives))
    except Exception as e:
        print(f"(génération du rapport échouée: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
