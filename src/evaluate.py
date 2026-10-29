"""Test-set evaluation: macro metrics, per-class metrics and a confusion matrix (contract: AGENTS.md)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on Colab/Docker; must run before pyplot is imported
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"


def predict(model: nn.Module, loader: DataLoader, device, *, max_batches: int | None = None,
            criterion: nn.Module | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """(y_true, y_pred, mean loss over samples). The loss is NaN without a criterion."""
    model.eval()
    y_true, y_pred, total_loss, n = [], [], 0.0, 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * len(y)
            n += len(y)
            y_true.append(y.cpu())
            y_pred.append(logits.argmax(1).cpu())
    mean_loss = total_loss / n if criterion is not None and n else float("nan")
    return torch.cat(y_true).numpy(), torch.cat(y_pred).numpy(), mean_loss


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], run_name: str) -> dict:
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "run_name": run_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f1.mean()),
        "per_class": {
            cls: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, cls in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate(model: nn.Module, loader: DataLoader, class_names: list[str], run_name: str, *,
             reports_dir: Path = REPORTS_DIR, max_batches: int | None = None, device=None) -> dict:
    """Metrics on `loader`, written to reports_dir/exp_<run>.json and cm_<run>.png."""
    if device is None:
        device = next(model.parameters()).device
    y_true, y_pred, _ = predict(model, loader, device, max_batches=max_batches)
    result = metrics_dict(y_true, y_pred, class_names, run_name)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"exp_{run_name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"{run_name}: test macro F1 {result['f1_macro']:.3f}")
    fig.tight_layout()
    fig.savefig(reports_dir / f"cm_{run_name}.png", dpi=120)
    plt.close(fig)
    return result
