# For the defence: evidence and results

Copies of the files in `reports/`, renamed so each name says what it is. The originals in `reports/` are the source
of truth (written by the Colab training run, committed unchanged in `76872cc` on 2026-10-29); these copies are only for
showing at the defence. Live demo: https://smartwaste-aiaiai.streamlit.app (test photos: `demo_images/`).

| File | What it shows |
|---|---|
| `01_training_log_all_epochs_3_runs.md` | Every epoch of the 3 runs (train/val loss, val macro F1, best epoch) + consistency checks |
| `02_results_summary_3_runs_test_metrics.csv` | Final table: settings, params, training time, val and test metrics per run |
| `03_loss_and_val_f1_curves_3_runs.png` | Training curves of all 3 runs |
| `04_optimizer_comparison_adam_vs_sgd_curves.png` | Adam vs SGD on the baseline CNN, curves |
| `05_optimizer_comparison_adam_vs_sgd.csv` | Adam vs SGD, numbers |
| `06_confusion_matrix_modified_cnn_adam_BEST.png` | Confusion matrix on test of the best model (served in the app) |
| `07_confusion_matrix_baseline_cnn_adam.png` | Confusion matrix, baseline CNN + Adam |
| `08_confusion_matrix_baseline_cnn_sgd.png` | Confusion matrix, baseline CNN + SGD |
| `09`–`11_raw_training_history_*.json` | Raw per-epoch logs saved by `src/train.py` during training |
| `12`–`14_raw_test_metrics_*.json` | Raw test metrics saved by `src/evaluate.py` (per class, confusion matrix) |
| `15_dataset_class_distribution.png` | Images per class |
| `16_dataset_sample_images.png` | Example photos of each class |
| `17_dataset_removed_duplicates_examples.png` | Examples of near-duplicate photos removed before the split |
| `18_dataset_class_counts_per_split.csv` | Per class: raw, duplicates removed, train/val/test counts |
| `19_dataset_duplicate_pairs_list.csv` | Every removed duplicate pair |

If `reports/` changes (e.g. a retrain), copy the files here again.
