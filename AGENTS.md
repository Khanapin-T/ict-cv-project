# AGENTS.md: SmartWaste AI

These are the shared rules for everyone working in this repo, people and coding agents alike (Codex, Claude Code).
`CLAUDE.md` only imports this file, so change the rules here.

## Project

SmartWaste AI classifies photos of waste into the 6 TrashNet classes
`cardboard, glass, metal, paper, plastic, trash`. Assignment option 1: the open dataset stays fixed and only
the ML algorithm changes. Exactly 3 runs, all on the same fixed split with the same seed:
`baseline_cnn` + Adam, `modified_cnn` (the baseline plus one extra conv block) + Adam, and `baseline_cnn` + SGD.
**The main comparison metric is macro F1.** The best of the 3 (by val macro F1) is served in a Streamlit app
(Docker). No augmentation, no synthetic or extra images, no pretrained models.

## Data rules (do not break)

- **`splits/splits.csv` is the single source of truth** for train/val/test. It was created once by
  `src/prepare_split.py` (perceptual-hash dedup, stratified 70/15/15, `seed=42`).
  **Never re-split**: don't sample your own split, don't edit the CSV by hand, and don't run
  `prepare_split.py --force`. Open an issue first if the split really has to change.
  - Columns: `path,label,split`. `path` is relative to the repo root with forward slashes
    (`data/raw/<class>/<file>.jpg`), `label` is a class name, and `split` is one of `train|val|test`.
  - Raw images in `data/raw/<class>/` are **not committed**. To get them on a fresh clone, run
    `python src/prepare_split.py`. It downloads them and checks them against the existing split
    without changing the split.
- Train, val and test use only the images listed in `splits.csv`, all with the eval transform below.
- Report metrics on **test**. Choose checkpoints and hyperparameters on **val** only.

## Interface contract

These conventions apply to all code:

- **Classes** are in alphabetical order: `["cardboard", "glass", "metal", "paper", "plastic", "trash"]`.
  A label index is the position in this list.
- **Input** is a `3x224x224` tensor. The eval transform (train, val, test **and** the app) is
  `Resize((224, 224))` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **Seed:** every run uses `seed=42` (`random`, `numpy`, `torch`, cuDNN deterministic, and the train
  DataLoader's shuffle generator), so runs differ only in what is being compared.
- **Main metric:** macro F1. Use it on val to pick checkpoints and on test to compare runs.
- **Run names:** use one `run_name` per training run for all of its files: `baseline_cnn`, `modified_cnn`,
  `baseline_cnn_sgd`.
  Never use `best`, which would overwrite the final model.

### `src/dataset.py`

`get_dataloaders(batch_size, *, num_workers=0, seed=42) -> (train_loader, val_loader, test_loader)`

Only the train loader shuffles, with a `torch.Generator` seeded by `seed`.

### `src/models.py`

`get_model(name, num_classes=6, dropout=0.0) -> nn.Module`

| `name` | Model |
|---|---|
| `"baseline_cnn"` | baseline CNN |
| `"modified_cnn"` | baseline CNN + one extra conv block |

Both are trained from scratch.
- `dropout` must not change the state_dict keys. Always create the `nn.Dropout(dropout)` layer
  (`p=0` is a no-op) so that a checkpoint loads whatever dropout it was trained with.
- An unknown `name` raises `ValueError`.

### `src/train.py`

`train(model_name, epochs, lr, optimizer, dropout, run_name, *, batch_size=32, max_batches=None, seed=42, ...)`,
with a CLI that exposes the same options. `optimizer` is `"adam"` or `"sgd"` (momentum 0.9). `max_batches` is
for smoke runs only; results of such runs (QUICK mode) are never committed. Each run saves:

- `models/<run>.pth`: `model.state_dict()`. This file stays local (gitignored).
- `reports/history_<run>.json`: `run_name`, `model_name`, `num_params`, `train_time_sec`, and the
  per-epoch lists `train_loss`, `val_loss` and `val_f1_macro`, plus `best_epoch` (1-based),
  `best_val_f1_macro` and `hparams` (the run's settings).

### `src/evaluate.py`

`evaluate(model, loader, class_names, run_name) -> dict` returns the metrics and writes them to:

- `reports/exp_<run>.json`: `run_name`, `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
  `per_class` (`{class: {"precision", "recall", "f1", "support"}}`) and `confusion_matrix`.
- `reports/cm_<run>.png`: the confusion matrix (rows are true labels, columns are predicted labels,
  in class order).

### Final model (read by the app)

- `models/best.pth`: the state_dict of the chosen run.
- `models/config.json`: `{"arch": "<get_model name>", "classes": [...], "run_name": "<run>"}`.
  This replaces the old `classes.json`.

## Defence topics

| Member | Topic |
|---|---|
| Participant 1 | Data: dataset & splits, data analysis, evaluation |
| Participant 2 | Models & training: baseline/modified CNN, training loop, results |
| Tair | Optimization (Adam vs SGD), app, Docker |

Tair's agent writes all code. Teammates prepare the slides for their topics.

## Git workflow

- Commits go straight to `main`.
- Never commit `data/raw/`, `.venv/` or other large generated files. In `models/`, only `best.pth`
  and `config.json` can be committed. Per-run checkpoints (`models/<run>.pth`) stay local.
- `models/best.pth` + `models/config.json` are produced by `notebooks/colab_train.ipynb` (`export_best`).
  Commit them together, after the results tables in `README.md` have been updated for the run they come from.
- Keep commits small and focused. Write messages in the imperative mood ("Add val loader").

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                  # versions are pinned; don't bump them casually
python src/prepare_split.py                      # download data/raw and verify it against splits.csv
streamlit run app/app.py                         # demo app on http://localhost:8501
pytest -q                                        # contract tests (CPU, tiny subset)
```
