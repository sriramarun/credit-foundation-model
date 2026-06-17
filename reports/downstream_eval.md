# Downstream Evaluation Report

> **Status:** TEMPLATE — Phase 5 deliverable (target day 43).

Compare three feature sets per task: **raw**, **embeddings-only**, **raw + embeddings**.

| Task | Feature set | Model | ROC-AUC | PR-AUC | KS | Gini | Brier | Lift@5% | Calibration |
|------|-------------|-------|--------:|-------:|---:|-----:|------:|--------:|-------------|
| Default 3M | Raw features | XGBoost | | | | | | | |
| Default 3M | Embeddings only | XGBoost | | | | | | | |
| Default 3M | Raw + embeddings | XGBoost | | | | | | | |

## Stability
- Lift and calibration by vintage and segment.

## Verdict
- Do embeddings show lift, or is there a clear failure diagnosis?
