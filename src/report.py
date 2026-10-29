"""Results across runs: summary.csv, loss curves, the exported best model and the README tables."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import CLASSES
from src.evaluate import REPORTS_DIR
from src.train import MODELS_DIR

SUMMARY_COLUMNS = ["run_name", "model_name", "optimizer", "lr", "dropout", "aug", "synthetic", "n_synthetic",
                   "epochs", "best_epoch", "num_params", "train_time_sec", "best_val_f1_macro", "accuracy",
                   "precision_macro", "recall_macro", "f1_macro", "trash_f1"]


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary_row(run_name: str, reports_dir: Path = REPORTS_DIR) -> dict:
    """One run's history_<run>.json joined with its exp_<run>.json (test metrics)."""
    reports_dir = Path(reports_dir)
    history = load_json(reports_dir / f"history_{run_name}.json")
    exp = load_json(reports_dir / f"exp_{run_name}.json")
    hp = history["hparams"]
    return {
        "run_name": run_name, "model_name": history["model_name"], "optimizer": hp["optimizer"], "lr": hp["lr"],
        "dropout": hp["dropout"], "aug": hp["aug"], "synthetic": hp["synthetic"], "n_synthetic": hp["n_synthetic"],
        "epochs": hp["epochs"], "best_epoch": history["best_epoch"], "num_params": history["num_params"],
        "train_time_sec": history["train_time_sec"], "best_val_f1_macro": history["best_val_f1_macro"],
        "accuracy": exp["accuracy"], "precision_macro": exp["precision_macro"], "recall_macro": exp["recall_macro"],
        "f1_macro": exp["f1_macro"], "trash_f1": exp["per_class"]["trash"]["f1"],
    }


def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(reports_dir: Path = REPORTS_DIR) -> list[dict]:
    """Every run that has both history_<run>.json and exp_<run>.json, written to reports_dir/summary.csv."""
    reports_dir = Path(reports_dir)
    runs = sorted(p.stem.removeprefix("history_") for p in reports_dir.glob("history_*.json"))
    rows = [summary_row(run, reports_dir) for run in runs if (reports_dir / f"exp_{run}.json").exists()]
    write_csv(rows, reports_dir / "summary.csv", SUMMARY_COLUMNS)
    return rows


def plot_loss_curves(run_names: list[str], out_path: Path, *, reports_dir: Path = REPORTS_DIR) -> None:
    """Train loss (solid), val loss (dashed) and val macro F1 per epoch, one color per run."""
    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(13, 5))
    for i, run in enumerate(run_names):
        path = Path(reports_dir) / f"history_{run}.json"
        if not path.exists():
            continue
        h = load_json(path)
        epochs = range(1, len(h["train_loss"]) + 1)
        color = f"C{i % 10}"
        ax_loss.plot(epochs, h["train_loss"], color=color, label=f"{run} train")
        ax_loss.plot(epochs, h["val_loss"], color=color, linestyle="--", label=f"{run} val")
        ax_f1.plot(epochs, h["val_f1_macro"], color=color, label=run)
    ax_loss.set(title="Loss (solid: train, dashed: val)", xlabel="epoch", ylabel="cross-entropy")
    ax_f1.set(title="Val macro F1", xlabel="epoch", ylabel="macro F1")
    for ax in (ax_loss, ax_f1):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def export_best(run_names: list[str], *, reports_dir: Path = REPORTS_DIR, models_dir: Path = MODELS_DIR) -> str:
    """Copy the run with the highest **val** macro F1 to models/best.pth and write models/config.json."""
    histories = {run: load_json(Path(reports_dir) / f"history_{run}.json") for run in run_names
                 if (Path(reports_dir) / f"history_{run}.json").exists()}
    if not histories:
        raise FileNotFoundError(f"None of {run_names} has a history_<run>.json in {reports_dir}")
    best = max(histories, key=lambda run: histories[run]["best_val_f1_macro"])
    models_dir = Path(models_dir)
    shutil.copyfile(models_dir / f"{best}.pth", models_dir / "best.pth")
    config = {"arch": histories[best]["model_name"], "classes": CLASSES, "run_name": best}
    (models_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return best


def _fmt(row: dict | None, key: str, fmt) -> str:
    return fmt(row[key]) if row else "—"


def results_markdown(rows: list[dict], arch_runs: list[str], data_runs: dict[str, str]) -> str:
    """README tables. A: architectures (`arch_runs`); B: data variants ({variant label: run name}).

    Runs missing from `rows` (not trained yet) still get a row, filled with "—".
    """
    by_run = {r["run_name"]: r for r in rows}
    f3 = "{:.3f}".format
    lines = ["**A) Architectures** (test set; best epoch picked on val macro F1)", "",
             "| Model | Run | Macro F1 | Accuracy | Params | Train time |", "|---|---|---|---|---|---|"]
    for run in arch_runs:
        r = by_run.get(run)
        lines.append(f"| {r['model_name'] if r else run} | {run} | {_fmt(r, 'f1_macro', f3)} | "
                     f"{_fmt(r, 'accuracy', f3)} | {_fmt(r, 'num_params', lambda v: f'{v / 1e6:.2f}M')} | "
                     f"{_fmt(r, 'train_time_sec', lambda v: f'{v / 60:.1f} min')} |")
    lines += ["", "**B) Data variants** (mobilenet_v2, test set)", "",
              "| Variant | Run | Macro F1 | Accuracy | Trash F1 |", "|---|---|---|---|---|"]
    for variant, run in data_runs.items():
        r = by_run.get(run)
        lines.append(f"| {variant} | {run} | {_fmt(r, 'f1_macro', f3)} | {_fmt(r, 'accuracy', f3)} | "
                     f"{_fmt(r, 'trash_f1', f3)} |")
    return "\n".join(lines) + "\n"
