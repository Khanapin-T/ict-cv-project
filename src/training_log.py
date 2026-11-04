"""Readable training log from the saved run reports: reports/training_log.md.

Reads only what the Colab training runs wrote (reports/history_<run>.json, reports/exp_<run>.json) and renders the
per-epoch losses and val macro F1 of every run, plus consistency checks that tie the numbers together.
Nothing is retrained or recomputed from images.

    python src/training_log.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
RUNS = ("baseline_cnn", "modified_cnn", "baseline_cnn_sgd")  # the 3 option-1 runs (AGENTS.md)
OUT_PATH = REPORTS_DIR / "training_log.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checks(history: dict, exp: dict) -> list[tuple[str, bool]]:
    """Checks that the saved numbers agree with each other and with the training contract."""
    f1 = history["val_f1_macro"]
    epochs = history["hparams"]["epochs"]
    best = max(range(len(f1)), key=lambda i: f1[i]) + 1  # first epoch with the top val F1, 1-based
    cm = exp["confusion_matrix"]
    support = [exp["per_class"][c]["support"] for c in exp["per_class"]]
    return [
        (f"{epochs} epochs logged for train loss, val loss and val F1",
         len(history["train_loss"]) == len(history["val_loss"]) == len(f1) == epochs),
        (f"best_epoch ({history['best_epoch']}) is the epoch with the highest val macro F1",
         history["best_epoch"] == best),
        ("best_val_f1_macro equals the max of the val F1 curve", abs(history["best_val_f1_macro"] - f1[best - 1]) < 1e-12),
        ("full run, not a smoke run (max_batches is null)", history["hparams"].get("max_batches") is None),
        ("seed is 42", history["hparams"].get("seed") == 42),
        ("confusion matrix row sums equal the per-class test support", [sum(r) for r in cm] == support),
        ("test accuracy equals the confusion matrix diagonal / total",
         abs(exp["accuracy"] - sum(cm[i][i] for i in range(len(cm))) / sum(map(sum, cm))) < 1e-12),
    ]


def run_section(run: str) -> tuple[str, bool]:
    history = load(REPORTS_DIR / f"history_{run}.json")
    exp = load(REPORTS_DIR / f"exp_{run}.json")
    hp = history["hparams"]
    results = checks(history, exp)
    lines = [
        f"## `{run}`",
        "",
        f"Model `{history['model_name']}`, {history['num_params']:,} parameters, optimizer {hp['optimizer']} "
        f"(lr {hp['lr']}), dropout {hp['dropout']}, batch {hp['batch_size']}, seed {hp['seed']}, "
        f"{hp['epochs']} epochs, training time {history['train_time_sec']:.1f} s.",
        f"Best epoch **{history['best_epoch']}** (val macro F1 {history['best_val_f1_macro']:.4f}); that checkpoint "
        f"scored test macro F1 **{exp['f1_macro']:.4f}**, accuracy {exp['accuracy']:.4f}.",
        "",
        "| Epoch | Train loss | Val loss | Val macro F1 |",
        "|---:|---:|---:|---:|",
    ]
    for i, (tl, vl, f1) in enumerate(zip(history["train_loss"], history["val_loss"], history["val_f1_macro"]), 1):
        mark = " ← best" if i == history["best_epoch"] else ""
        lines.append(f"| {i}{mark} | {tl:.4f} | {vl:.4f} | {f1:.4f} |")
    lines += ["", "Consistency checks:", ""]
    lines += [f"- {'✅' if ok else '❌'} {text}" for text, ok in results]
    return "\n".join(lines) + "\n", all(ok for _, ok in results)


def main() -> int:
    sections, all_ok = [], True
    for run in RUNS:
        text, ok = run_section(run)
        sections.append(text)
        all_ok &= ok
    header = (
        "# Training log\n\n"
        "Per-epoch numbers of the 3 runs, as saved by `src/train.py` during the Colab training "
        "(`notebooks/colab_train.ipynb`) in `reports/history_<run>.json`, with the test metrics from "
        "`reports/exp_<run>.json`. Written by `python src/training_log.py`: nothing here is retyped by hand.\n"
    )
    OUT_PATH.write_text(header + "\n" + "\n".join(sections), encoding="utf-8")
    print(f"Wrote {OUT_PATH}; all checks {'passed' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
