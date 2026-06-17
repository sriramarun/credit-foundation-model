# Model Card — Credit-TFM

> **Status:** TEMPLATE — Phase 8 deliverable (target day 59).

## Model details
- **Name / version:** Credit-TFM-_TBD_
- **Architecture:** Decoder foundation model (NVIDIA TFM blueprint lineage)
- **Parameters / context:** _e.g. 100M–250M, 4k–8k_
- **Tokenizer:** _v1 / v2, config reference_
- **Training window / compute:** _8× H100, dates_

## Intended use
- **Primary:** Self-supervised credit embeddings for downstream risk models
  (default/delinquency, prepayment, cure, segmentation, anomaly detection).
- **Out of scope:** Standalone automated credit decisions without human review;
  use on populations outside the training distribution.

## Training data
- Summary and provenance: see [`data_card.md`](data_card.md).

## Evaluation
- Downstream lift vs raw-feature baselines: see
  [`downstream_eval.md`](downstream_eval.md) and the `Metrics` roadmap tab.
- Report ROC-AUC, PR-AUC, KS, Gini, Brier, Lift@5%, and **calibration separately**.

## Limitations & risks
- Calibration may degrade even when ranking improves — recalibrate downstream models.
- Context truncation may drop long-horizon behavior — see context ablations.
- Leakage audit summary and residual risks: _link Phase 6 audit_.

## Ethical / fairness considerations
- _Document protected-attribute handling, disparate-impact checks, and review process._
