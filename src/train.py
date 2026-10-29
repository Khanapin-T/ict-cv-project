"""Train one run: pick the best epoch on val macro F1, then evaluate it on test (contract: AGENTS.md).

Example:
    python src/train.py --model mobilenet_v2 --epochs 15 --lr 1e-4 --dropout 0.2 --aug --run-name mobilenet_v2_aug
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `python src/train.py` puts src/ on sys.path, not the repo root

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.dataset import CLASSES, get_dataloaders
from src.evaluate import REPORTS_DIR, evaluate, predict
from src.models import MODEL_NAMES, count_params, get_model

MODELS_DIR = ROOT / "models"
OPTIMIZERS = ("adam", "sgd")


def check_run_name(run_name: str) -> None:
    """`best` (any case: Windows/macOS file names ignore it) would overwrite the final model; a path separator would
    write outside models/ and reports/."""
    if not run_name or run_name.lower() == "best" or "/" in run_name or "\\" in run_name:
        raise ValueError(f"Invalid run_name {run_name!r}: must not be 'best' or contain '/' or '\\'")


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


def train(model_name: str, epochs: int, lr: float, optimizer: str, dropout: float, aug: bool, synthetic: bool,
          run_name: str, *, batch_size: int = 32, max_batches: int | None = None, pretrained: bool = True,
          num_workers: int | None = None, seed: int = 42, device=None, models_dir: Path = MODELS_DIR,
          reports_dir: Path = REPORTS_DIR) -> dict:
    """Train, keep the best val-macro-F1 epoch in models_dir/<run>.pth, return its test metrics."""
    check_run_name(run_name)
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {optimizer!r}; expected one of {OPTIMIZERS}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size, aug, synthetic, num_workers=num_workers,
                                                            seed=seed)
    model = get_model(model_name, num_classes=len(CLASSES), dropout=dropout, pretrained=pretrained).to(device)
    opt = make_optimizer(optimizer, model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"{run_name}.pth"

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": []}
    best_f1, best_epoch = -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        y_true, y_pred, val_loss = predict(model, val_loader, device, max_batches=max_batches, criterion=criterion)
        val_f1 = float(f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0))
        history["train_loss"].append(total / n)
        history["val_loss"].append(val_loss)
        history["val_f1_macro"].append(val_f1)
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            torch.save(model.state_dict(), ckpt)
        print(f"[{run_name}] epoch {epoch}/{epochs}  train_loss {total / n:.4f}  val_loss {val_loss:.4f}  "
              f"val_f1 {val_f1:.4f}{'  *' if best_epoch == epoch else ''}", flush=True)
    train_time = time.perf_counter() - start

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    result = evaluate(model, test_loader, CLASSES, run_name, reports_dir=reports_dir, max_batches=max_batches,
                      device=device)
    history = {
        "run_name": run_name,
        "model_name": model_name,
        "num_params": count_params(model),
        "train_time_sec": round(train_time, 1),
        **history,
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_f1,
        "hparams": {"epochs": epochs, "lr": lr, "optimizer": optimizer, "dropout": dropout, "aug": aug,
                    "synthetic": synthetic, "n_synthetic": train_loader.dataset.n_synthetic,
                    "batch_size": batch_size, "seed": seed, "max_batches": max_batches},
    }
    (reports_dir / f"history_{run_name}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[{run_name}] best epoch {best_epoch} (val F1 {best_f1:.4f}); test F1 {result['f1_macro']:.4f}, "
          f"accuracy {result['accuracy']:.4f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--aug", action="store_true", help="augment the train set")
    parser.add_argument("--synthetic", action="store_true", help="add data/extra/synthetic/ to the train set")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-batches", type=int, default=None, help="limit every loop (smoke runs)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true", help="mobilenet_v2 without ImageNet weights")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    a = parser.parse_args()
    train(a.model, a.epochs, a.lr, a.optimizer, a.dropout, a.aug, a.synthetic, a.run_name,
          batch_size=a.batch_size, max_batches=a.max_batches, pretrained=not a.no_pretrained,
          models_dir=a.models_dir, reports_dir=a.reports_dir)


if __name__ == "__main__":
    main()
