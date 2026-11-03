# Progress

## Goal
SmartWaste AI: classify waste photos into 6 TrashNet classes (cardboard, glass, metal, paper, plastic, trash).
Assignment option 1: fixed dataset and split, only the ML algorithm changes. Exactly 3 runs, seed 42:
baseline_cnn + Adam, modified_cnn (baseline + one conv block) + Adam, baseline_cnn + SGD.
Main metric: macro F1. Status: project finished. Team is now studying the code and results for the defence.
Docker: agreed out of scope, not built or run. Do not install it.

## Done
- Split: `splits/splits.csv` (perceptual-hash dedup, stratified 70/15/15, seed 42). Never re-split.
- Code: `src/dataset.py`, `models.py`, `train.py` (CLI), `evaluate.py`, `optimization.py`, `report.py`,
  `prepare_split.py`. Streamlit app in `app/app.py`.
- Notebooks: `notebooks/data_analysis.ipynb` (executed), `notebooks/colab_train.ipynb` (ran the 3 runs on Colab).
- Tests: `tests/test_contract.py`, `test_app.py`, `test_notebooks.py`.
- Results committed (`76872cc`): reports/, models/best.pth, models/config.json, README tables.

## Results (test set; checkpoint chosen on val)

| Run | Val macro F1 | Test macro F1 | Test accuracy | Params |
|---|---|---|---|---|
| baseline_cnn (Adam) | 0.813 | 0.768 | 0.781 | 0.62M |
| modified_cnn (Adam) | 0.817 | 0.800 | 0.813 | 1.44M |
| baseline_cnn_sgd (SGD) | 0.781 | 0.792 | 0.810 | 0.62M |

Best by val macro F1: modified_cnn. It is served as `models/best.pth` (config: `models/config.json`).
All runs: 30 epochs, dropout 0.3, batch 32.

## Next steps
- Team: understand the pipeline and results, and prepare defence material per topic (AGENTS.md table).
- Optional: fix the deferred minors below if the team wants them.
- Not planned: Docker build, deploy.

## Key decisions
- Dedup: 256-bit phash, threshold 48; the same photo under two labels is dropped from both.
- Eval transform lives only in `src/dataset.py`; the app imports it.
- Best epoch and best run are picked on val macro F1; test is only reported.
- `export_best` picks the winner by val; the winner's checkpoint must exist locally.
- SGD run reuses the Part A baseline_cnn (Adam) and trains only baseline_cnn_sgd (lr 1e-2, momentum 0.9).
- Run names: `baseline_cnn`, `modified_cnn`, `baseline_cnn_sgd`. Never `best`.
- Colab: QUICK runs cannot be pushed; the token is never written to logs.

## Important files
AGENTS.md (rules), README.md (results + how to train), src/*.py, tests/, app/app.py,
notebooks/colab_train.ipynb, notebooks/data_analysis.ipynb, splits/splits.csv, models/best.pth,
models/config.json, reports/summary.csv, docs/plans/full-pipeline.md (original plan, local only)

## Gotchas
- `prepare_split.py` without `--force` only verifies the split; `--force` must never be used.
- `data/raw/` is not committed; run `python src/prepare_split.py` on a fresh clone to fetch it.
- Deferred minors from review: `epochs=0` gives ZeroDivision; I;16 PNGs are not handled;
  `optimization` checks the experiment file late; notebook edge cases with partial clones and no-change commits;
  `test_app` skips once a real best.pth exists.
- Bash heredocs collapse backslashes; use Edit for text with `\`.
- `AppTest.from_file` needs an absolute path.
- Git: `PROGRESS.md` and `docs/` are untracked (local). Commit only when asked. No co-author lines.
