# AGENTS.md: SmartWaste AI

These are the shared rules for everyone working in this repo, people and coding agents alike (Codex, Claude Code).
`CLAUDE.md` only imports this file, so change the rules here.

## Project

SmartWaste AI classifies photos of waste into the 6 TrashNet classes
`cardboard, glass, metal, paper, plastic, trash`. We compare a baseline CNN, a modified CNN (the baseline
plus one extra conv block) and an ImageNet-pretrained MobileNetV2, with and without augmentation and
synthetic data, all on the same fixed split. **The main comparison metric is macro F1.** The best model
is served in a Streamlit app (Docker).

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
- **Augmentation and `data/extra/synthetic/` are TRAIN ONLY.** Val and test use only the real images
  listed in `splits.csv`, with the eval transform below. Never augment val/test, and never put
  synthetic images into val/test.
- Synthetic images go in `data/extra/synthetic/<class>/`, where the folder name is the label. They
  **are committed**. Never put real TrashNet images there. If a synthetic image is derived from a real
  one (img2img, edits), the source must be a **train** image, never val/test (that would leak).
- Report metrics on **test**. Choose checkpoints and hyperparameters on **val** only.

## Interface contract

These conventions apply to all code:

- **Classes** are in alphabetical order: `["cardboard", "glass", "metal", "paper", "plastic", "trash"]`.
  A label index is the position in this list.
- **Input** is a `3x224x224` tensor. The eval transform (val/test **and** the app) is
  `Resize((224, 224))` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
  Augmentation is added to the train transform only.
- **Main metric:** macro F1. Use it on val to pick checkpoints and on test to compare runs.
- **Run names:** use one `run_name` per training run for all of its files (e.g. `mobilenet_v2_aug`).
  Never use `best`, which would overwrite the final model.

### `src/dataset.py`

`get_dataloaders(batch_size, aug: bool, synthetic: bool) -> (train_loader, val_loader, test_loader)`

`aug=True` augments the train set only. `synthetic=True` adds `data/extra/synthetic/<class>/*` to the
train set only.

### `src/models.py`

`get_model(name, num_classes=6, dropout=0.0, *, pretrained=True) -> nn.Module`

| `name` | Model |
|---|---|
| `"baseline_cnn"` | baseline CNN |
| `"modified_cnn"` | baseline CNN + one extra conv block |
| `"mobilenet_v2"` | torchvision MobileNetV2, ImageNet pretrained, new classifier head |

- `pretrained=False` skips the ImageNet download. The app uses it because it loads `models/best.pth`
  anyway. The CNNs accept the argument and ignore it.
- `dropout` must not change the state_dict keys. Always create the `nn.Dropout(dropout)` layer
  (`p=0` is a no-op) so that a checkpoint loads whatever dropout it was trained with.
- An unknown `name` raises `ValueError`.

### `src/train.py`

`train(model_name, epochs, lr, optimizer, dropout, aug, synthetic, run_name, ...)`, with a CLI that
exposes the same options. Each run saves:

- `models/<run>.pth`: `model.state_dict()`. This file stays local (gitignored).
- `reports/history_<run>.json`: `run_name`, `model_name`, `num_params`, `train_time_sec`, and the
  per-epoch lists `train_loss`, `val_loss` and `val_f1_macro`.

### `src/evaluate.py`

`evaluate(model, loader, class_names, run_name) -> dict` returns the metrics and writes them to:

- `reports/exp_<run>.json`: `run_name`, `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`,
  and `per_class` (`{class: {"precision", "recall", "f1", "support"}}`).
- `reports/cm_<run>.png`: the confusion matrix (rows are true labels, columns are predicted labels,
  in class order).

### Final model (read by the app)

- `models/best.pth`: the state_dict of the chosen run.
- `models/config.json`: `{"arch": "<get_model name>", "classes": [...], "run_name": "<run>"}`.
  This replaces the old `classes.json`.

## File ownership

| Owner | Files |
|---|---|
| Participant 1 | `src/dataset.py`, `src/evaluate.py`, `notebooks/data_analysis.ipynb`, `data/extra/synthetic/`, `reports/data_*` |
| Participant 2 | `src/models.py`, `src/train.py`, `notebooks/colab_train.ipynb`, results table in `README.md`, `models/best.pth` + `models/config.json` |
| Tair | `src/optimization.py`, `app/`, `Dockerfile`, `.dockerignore`, `README.md` (everything except the results table) |
| Shared: change only via a PR the whole team has seen | `AGENTS.md`, `requirements.txt`, `src/__init__.py`, `src/prepare_split.py`, `splits/` |

Agents: edit only files owned by the person you are working for. If a change is needed in someone
else's file, describe it in the PR instead of making it.

## Git workflow

- Work on your own branch: `data` (Participant 1), `model` (Participant 2), `app` (Tair).
- Merge through a **PR to `main`**. **No direct pushes to `main`.**
- Never commit `data/raw/`, `.venv/` or other large generated files. In `models/`, only `best.pth`
  and `config.json` can be committed. Per-run checkpoints (`models/<run>.pth`) stay local.
- **Only Participant 2 commits `models/best.pth` and `models/config.json`.** Commit them together, in
  a PR, and only after the results table in `README.md` has been updated for the run they come from.
  Nobody else commits model files.
- Keep commits small and focused. Write messages in the imperative mood ("Add val loader").

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                  # versions are pinned; don't bump them casually
python src/prepare_split.py                      # download data/raw and verify it against splits.csv
streamlit run app/app.py                         # demo app on http://localhost:8501
```
