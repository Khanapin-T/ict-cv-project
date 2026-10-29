"""Optimizer comparison on baseline_cnn: Adam vs SGD (momentum 0.9).

Adam uses lr 1e-3 and SGD (momentum 0.9) uses lr 1e-2. Each optimizer uses its own standard learning rate,
because SGD needs a larger step than Adam. So the comparison is between the two optimizers as typically
configured, not at an identical lr.

The Adam side is Part A's `baseline_cnn` run, reused as is (like Part B reuses `mobilenet_v2`); only
`baseline_cnn_sgd` is trained here, with the Adam run's epochs, dropout, aug, synthetic, batch size, seed and
max_batches.

Example:
    python src/optimization.py               # after Part A's baseline_cnn run
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `python src/optimization.py` puts src/ on sys.path, not the repo root

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import REPORTS_DIR
from src.report import load_json, summary_row, write_csv
from src.train import MODELS_DIR, train

OPTIMIZATION_COLUMNS = ["run_name", "optimizer", "lr", "best_epoch", "best_val_f1_macro", "f1_macro", "accuracy",
                        "train_time_sec"]


def compare_optimizers(adam_run: str = "baseline_cnn", lr_sgd: float = 1e-2, *, reports_dir: Path = REPORTS_DIR,
                       models_dir: Path = MODELS_DIR, **train_kwargs) -> list[dict]:
    """Train `<adam_run>_sgd` with the Adam run's settings, then write optimization.csv and the curves.

    `train_kwargs` go to train() and override the copied settings (e.g. max_batches for a smoke run).
    """
    reports_dir = Path(reports_dir)
    history_path = reports_dir / f"history_{adam_run}.json"
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found: run Part A's {adam_run} first "
                                "(the Adam side is reused, never retrained here)")
    adam = load_json(history_path)
    hp = adam["hparams"]
    sgd_run = f"{adam_run}_sgd"
    kwargs = {"batch_size": hp["batch_size"], "seed": hp["seed"], "max_batches": hp["max_batches"], **train_kwargs}
    train(adam["model_name"], hp["epochs"], lr_sgd, "sgd", hp["dropout"], hp["aug"], hp["synthetic"], sgd_run,
          models_dir=models_dir, reports_dir=reports_dir, **kwargs)

    runs = [adam_run, sgd_run]
    rows = [summary_row(run, reports_dir) for run in runs]
    write_csv(rows, reports_dir / "optimization.csv", OPTIMIZATION_COLUMNS)

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in rows:
        h = load_json(reports_dir / f"history_{row['run_name']}.json")
        epochs = range(1, len(h["train_loss"]) + 1)
        label = f"{row['optimizer']} (lr {row['lr']:g})"
        ax_loss.plot(epochs, h["train_loss"], label=label)
        ax_f1.plot(epochs, h["val_f1_macro"], label=label)
    ax_loss.set(title="Train loss", xlabel="epoch", ylabel="cross-entropy")
    ax_f1.set(title="Val macro F1", xlabel="epoch", ylabel="macro F1")
    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{adam['model_name']}: Adam (lr {rows[0]['lr']:g}) vs SGD momentum 0.9 (lr {lr_sgd:g})")
    fig.tight_layout()
    fig.savefig(reports_dir / "optimization_curves.png", dpi=120)
    plt.close(fig)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adam-run", default="baseline_cnn")
    parser.add_argument("--lr-sgd", type=float, default=1e-2)
    parser.add_argument("--max-batches", type=int, default=None, help="override the Adam run's max_batches")
    a = parser.parse_args()
    overrides = {} if a.max_batches is None else {"max_batches": a.max_batches}
    for row in compare_optimizers(a.adam_run, a.lr_sgd, **overrides):
        print(row)


if __name__ == "__main__":
    main()
