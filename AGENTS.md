# AGENTS.md: SmartWaste AI

These are the shared rules for everyone working in this repo, people and coding agents alike (Codex, Claude Code).
`CLAUDE.md` only imports this file, so change the rules here.

## Project

SmartWaste AI classifies photos of waste into the 6 TrashNet classes
`cardboard, glass, metal, paper, plastic, trash` with a fine-tuned **MobileNetV2**
(torchvision, ImageNet weights). We compare 4 training setups (original / +aug / +synthetic / +both)
on the same fixed split and serve the best model in a Streamlit app (Docker).

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

| Item | Contract |
|---|---|
| `src/dataset.py` | `get_dataloaders(batch_size: int, aug: bool, synthetic: bool) -> (train_loader, val_loader, test_loader)` |
| Labels | index = position in `["cardboard", "glass", "metal", "paper", "plastic", "trash"]` (sorted class names) |
| Eval transform (val/test **and** app) | `Resize((224, 224))` → `ToTensor()` → `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` |
| `aug=True` | adds random augmentation to the **train** transform only |
| `synthetic=True` | appends `data/extra/synthetic/<class>/*` to the **train** set only |
| Model | `torchvision.models.mobilenet_v2`, with `classifier[1] = nn.Linear(1280, 6)` |
| `models/best.pth` | `torch.save(model.state_dict(), ...)`: a plain state_dict of the checkpoint with the best val score |
| `models/classes.json` | JSON list of class names in label-index order |
| `reports/<run_name>.json` | test-set metrics, schema below |

Run names are `original`, `aug`, `synthetic` and `aug_synthetic`.

`reports/<run_name>.json` uses this schema (precision/recall/f1 are **macro** averages):

```json
{
  "run_name": "aug",
  "split": "test",
  "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
  "accuracy": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "per_class": {"cardboard": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}},
  "confusion_matrix": [[0, 0, 0, 0, 0, 0]]
}
```

`per_class` has one entry per class. `confusion_matrix` is 6×6: rows are true labels, columns are
predicted labels, in `classes` order.

## File ownership

| Area | Owner |
|---|---|
| `src/dataset.py`, data analysis (`notebooks/`) | Participant 1 |
| `src/train.py`, `src/evaluate.py` | Participant 2 |
| `app/`, `Dockerfile`, `.dockerignore`, `README.md` | Tair |
| `AGENTS.md`, `requirements.txt`, `src/prepare_split.py`, `splits/`, `data/extra/synthetic/` | shared: change only via a PR the whole team has seen |

Agents: edit only files owned by the person you are working for. If a change is needed in someone
else's file, describe it in the PR instead of making it.

## Git workflow

- Work on your own branch: `data` (Participant 1), `model` (Participant 2), `app` (Tair).
- Merge through a **PR to `main`**. **No direct pushes to `main`.**
- Never commit `data/raw/`, `.venv/` or other large generated files. `*.pth` is gitignored.
- Keep commits small and focused. Write messages in the imperative mood ("Add val loader").

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt                  # versions are pinned; don't bump them casually
python src/prepare_split.py                      # download data/raw and verify it against splits.csv
streamlit run app/app.py                         # demo app on http://localhost:8501
```
