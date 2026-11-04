# Training log

Per-epoch numbers of the 3 runs, as saved by `src/train.py` during the Colab training (`notebooks/colab_train.ipynb`) in `reports/history_<run>.json`, with the test metrics from `reports/exp_<run>.json`. Written by `python src/training_log.py`: nothing here is retyped by hand.

## `baseline_cnn`

Model `baseline_cnn`, 619,782 parameters, optimizer adam (lr 0.001), dropout 0.3, batch 32, seed 42, 30 epochs, training time 241.0 s.
Best epoch **22** (val macro F1 0.8129); that checkpoint scored test macro F1 **0.7683**, accuracy 0.7807.

| Epoch | Train loss | Val loss | Val macro F1 |
|---:|---:|---:|---:|
| 1 | 1.3814 | 1.1520 | 0.6057 |
| 2 | 1.0297 | 1.1726 | 0.5570 |
| 3 | 1.0071 | 1.0599 | 0.5701 |
| 4 | 0.8719 | 1.0183 | 0.6191 |
| 5 | 0.8216 | 1.0110 | 0.5863 |
| 6 | 0.7474 | 1.1924 | 0.5629 |
| 7 | 0.7042 | 0.8461 | 0.6224 |
| 8 | 0.6696 | 0.8531 | 0.6821 |
| 9 | 0.6236 | 0.8513 | 0.6573 |
| 10 | 0.5372 | 0.7513 | 0.7148 |
| 11 | 0.5354 | 0.7580 | 0.7054 |
| 12 | 0.4744 | 0.6763 | 0.7728 |
| 13 | 0.4073 | 0.7378 | 0.7469 |
| 14 | 0.4398 | 1.0595 | 0.6559 |
| 15 | 0.4123 | 1.1800 | 0.6720 |
| 16 | 0.3678 | 0.7777 | 0.7381 |
| 17 | 0.3268 | 1.0464 | 0.6963 |
| 18 | 0.3358 | 0.7559 | 0.7466 |
| 19 | 0.3209 | 0.8637 | 0.7057 |
| 20 | 0.2743 | 0.8543 | 0.7665 |
| 21 | 0.2538 | 0.7315 | 0.7910 |
| 22 ← best | 0.2765 | 0.7096 | 0.8129 |
| 23 | 0.2323 | 0.7279 | 0.7890 |
| 24 | 0.2300 | 0.6806 | 0.8020 |
| 25 | 0.2265 | 0.9907 | 0.6994 |
| 26 | 0.2132 | 1.5414 | 0.6538 |
| 27 | 0.2062 | 0.9406 | 0.7492 |
| 28 | 0.2046 | 0.8312 | 0.7866 |
| 29 | 0.1775 | 0.8843 | 0.7343 |
| 30 | 0.1731 | 1.0796 | 0.6988 |

Consistency checks:

- ✅ 30 epochs logged for train loss, val loss and val F1
- ✅ best_epoch (22) is the epoch with the highest val macro F1
- ✅ best_val_f1_macro equals the max of the val F1 curve
- ✅ full run, not a smoke run (max_batches is null)
- ✅ seed is 42
- ✅ confusion matrix row sums equal the per-class test support
- ✅ test accuracy equals the confusion matrix diagonal / total

## `modified_cnn`

Model `modified_cnn`, 1,439,750 parameters, optimizer adam (lr 0.001), dropout 0.3, batch 32, seed 42, 30 epochs, training time 244.5 s.
Best epoch **30** (val macro F1 0.8175); that checkpoint scored test macro F1 **0.7997**, accuracy 0.8128.

| Epoch | Train loss | Val loss | Val macro F1 |
|---:|---:|---:|---:|
| 1 | 1.4677 | 1.1584 | 0.5237 |
| 2 | 1.0953 | 1.2327 | 0.5200 |
| 3 | 1.0239 | 1.0254 | 0.5489 |
| 4 | 0.9413 | 1.1774 | 0.4975 |
| 5 | 0.8837 | 1.2475 | 0.5145 |
| 6 | 0.7833 | 0.8361 | 0.6929 |
| 7 | 0.7416 | 0.9888 | 0.6183 |
| 8 | 0.6902 | 0.8577 | 0.6785 |
| 9 | 0.6722 | 1.1126 | 0.5843 |
| 10 | 0.6302 | 1.0434 | 0.6609 |
| 11 | 0.6188 | 0.8603 | 0.6668 |
| 12 | 0.5479 | 0.8159 | 0.6840 |
| 13 | 0.5208 | 0.8451 | 0.7047 |
| 14 | 0.5255 | 0.8020 | 0.7020 |
| 15 | 0.4933 | 0.7393 | 0.7183 |
| 16 | 0.4330 | 0.8439 | 0.6682 |
| 17 | 0.4073 | 0.9095 | 0.7016 |
| 18 | 0.3882 | 0.7936 | 0.6970 |
| 19 | 0.3811 | 0.8362 | 0.7226 |
| 20 | 0.3679 | 1.1130 | 0.7196 |
| 21 | 0.3207 | 0.8469 | 0.7505 |
| 22 | 0.3130 | 0.8822 | 0.7447 |
| 23 | 0.3039 | 0.9410 | 0.7276 |
| 24 | 0.2483 | 0.7434 | 0.7660 |
| 25 | 0.2953 | 0.7039 | 0.7781 |
| 26 | 0.2487 | 0.7367 | 0.7934 |
| 27 | 0.2810 | 0.7263 | 0.7943 |
| 28 | 0.2421 | 0.7103 | 0.7995 |
| 29 | 0.2480 | 0.7456 | 0.7942 |
| 30 ← best | 0.2443 | 0.6761 | 0.8175 |

Consistency checks:

- ✅ 30 epochs logged for train loss, val loss and val F1
- ✅ best_epoch (30) is the epoch with the highest val macro F1
- ✅ best_val_f1_macro equals the max of the val F1 curve
- ✅ full run, not a smoke run (max_batches is null)
- ✅ seed is 42
- ✅ confusion matrix row sums equal the per-class test support
- ✅ test accuracy equals the confusion matrix diagonal / total

## `baseline_cnn_sgd`

Model `baseline_cnn`, 619,782 parameters, optimizer sgd (lr 0.01), dropout 0.3, batch 32, seed 42, 30 epochs, training time 241.4 s.
Best epoch **25** (val macro F1 0.7806); that checkpoint scored test macro F1 **0.7925**, accuracy 0.8102.

| Epoch | Train loss | Val loss | Val macro F1 |
|---:|---:|---:|---:|
| 1 | 1.4014 | 1.2172 | 0.4011 |
| 2 | 1.1061 | 2.2738 | 0.2751 |
| 3 | 1.0396 | 1.0494 | 0.5859 |
| 4 | 0.9766 | 1.0839 | 0.5979 |
| 5 | 0.9294 | 0.9313 | 0.6154 |
| 6 | 0.8297 | 0.9040 | 0.6571 |
| 7 | 0.7681 | 0.8723 | 0.6543 |
| 8 | 0.7583 | 1.2998 | 0.5586 |
| 9 | 0.6769 | 0.9385 | 0.6680 |
| 10 | 0.6124 | 1.1972 | 0.6254 |
| 11 | 0.6184 | 0.7405 | 0.6851 |
| 12 | 0.5582 | 1.1209 | 0.6163 |
| 13 | 0.5077 | 0.8041 | 0.6973 |
| 14 | 0.4811 | 0.8781 | 0.6787 |
| 15 | 0.4552 | 0.7873 | 0.7048 |
| 16 | 0.4123 | 1.0132 | 0.7071 |
| 17 | 0.3819 | 0.8843 | 0.7003 |
| 18 | 0.3714 | 1.0254 | 0.6810 |
| 19 | 0.3349 | 1.0035 | 0.6866 |
| 20 | 0.3356 | 1.2832 | 0.6665 |
| 21 | 0.2791 | 1.1435 | 0.7113 |
| 22 | 0.3103 | 1.4686 | 0.6580 |
| 23 | 0.2466 | 0.7365 | 0.7597 |
| 24 | 0.2502 | 0.9917 | 0.6986 |
| 25 ← best | 0.2289 | 0.7925 | 0.7806 |
| 26 | 0.2001 | 1.1063 | 0.7305 |
| 27 | 0.2146 | 0.9244 | 0.7672 |
| 28 | 0.1984 | 0.9264 | 0.7504 |
| 29 | 0.1938 | 1.0259 | 0.7284 |
| 30 | 0.1749 | 0.8820 | 0.7619 |

Consistency checks:

- ✅ 30 epochs logged for train loss, val loss and val F1
- ✅ best_epoch (25) is the epoch with the highest val macro F1
- ✅ best_val_f1_macro equals the max of the val F1 curve
- ✅ full run, not a smoke run (max_batches is null)
- ✅ seed is 42
- ✅ confusion matrix row sums equal the per-class test support
- ✅ test accuracy equals the confusion matrix diagonal / total
