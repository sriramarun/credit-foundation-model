# Evaluation

Downstream tasks: `default_6m`, `default_3m`, `prepayment_6m`, `cure_3m`, `segmentation`.

Metrics: ROC-AUC, PR-AUC, KS, Gini, Brier, lift@K, calibration error. Compare four feature
regimes: XGBoost baseline / embeddings-only / raw+embeddings / LoRA fine-tuned. Calibration
is reported separately from ranking.
