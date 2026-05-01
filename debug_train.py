"""
Smoke-test : pour chaque modèle dans models_code/, vérifie que le pipeline
(dataset -> dataloader -> forward -> loss -> backward -> step -> eval)
tourne sans crash. Quelques itérations seulement, sur CPU.

Usage : uv run debug_train.py
"""

from pathlib import Path
import importlib
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from main import SmthSmthDataset, load_model, _to_device


NUM_CLASSES = 33
IMAGE_SIZE = 64           # plus petit que 112 pour aller vite
BATCH_SIZE = 2
N_TRAIN_BATCHES = 2
N_VAL_BATCHES = 1
NUM_WORKERS = 0           # 0 = plus simple à debugger
ARCHIVES_ROOT = "archives"


def smoke_test(model_name: str, device: torch.device) -> bool:
    print(f"\n=== {model_name} ===")
    t0 = time.perf_counter()

    model, input_mode = load_model(model_name, NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  INPUT_MODE = {input_mode} | params = {n_params:,}")

    full_train = SmthSmthDataset(ARCHIVES_ROOT, "train", input_mode, IMAGE_SIZE)
    full_val = SmthSmthDataset(ARCHIVES_ROOT, "val", input_mode, IMAGE_SIZE)
    print(f"  dataset train={len(full_train):,} val={len(full_val):,}")

    n_train = N_TRAIN_BATCHES * BATCH_SIZE
    n_val = N_VAL_BATCHES * BATCH_SIZE
    train_subset = Subset(full_train, list(range(min(n_train, len(full_train)))))
    val_subset = Subset(full_val, list(range(min(n_val, len(full_val)))))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train pass
    model.train()
    for i, (inputs, y) in enumerate(train_loader):
        inputs = _to_device(inputs, device)
        y = y.to(device)
        shapes = [tuple(t.shape) for t in inputs]
        print(f"  [train b{i}] input shapes={shapes} y={tuple(y.shape)}")
        optimizer.zero_grad()
        logits = model(*inputs)
        assert logits.shape == (y.size(0), NUM_CLASSES), \
            f"logits shape {logits.shape} != ({y.size(0)}, {NUM_CLASSES})"
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"  [train b{i}] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    # Eval pass
    model.eval()
    with torch.no_grad():
        for i, (inputs, y) in enumerate(val_loader):
            inputs = _to_device(inputs, device)
            y = y.to(device)
            logits = model(*inputs)
            loss = criterion(logits, y)
            print(f"  [val   b{i}] logits={tuple(logits.shape)} loss={loss.item():.4f}")

    print(f"  OK ({time.perf_counter() - t0:.1f}s)")
    return True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_dir = Path("models_code")
    model_names = sorted(p.stem for p in models_dir.glob("*.py") if p.stem != "__init__")
    print(f"Modèles détectés : {model_names}")

    failures: list[tuple[str, str]] = []
    for name in model_names:
        try:
            smoke_test(name, device)
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ÉCHEC : {type(e).__name__}: {e}")

    print("\n=== Résumé ===")
    if failures:
        for name, err in failures:
            print(f"  KO  {name}  -> {err}")
        raise SystemExit(1)
    print(f"  OK : tous les modèles passent ({len(model_names)})")


if __name__ == "__main__":
    main()
