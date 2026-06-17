# Baseline Benchmark Report

> **Status:** TEMPLATE — Phase 2 deliverable (target day 14). Gate: strong baseline + trustworthy labels.

## Setup
- Feature set: raw observation-date features
- Models: logistic regression, XGBoost, LightGBM
- Splits: temporal train / validation / test

## Results (fill the `Metrics` roadmap tab and mirror here)
| Task | Model | ROC-AUC | PR-AUC | KS | Gini | Brier | Lift@5% | Calibration |
|------|-------|--------:|-------:|---:|-----:|------:|--------:|-------------|
| Default 3M | XGBoost | | | | | | | |

## Data quality findings
- _Missingness, label integrity, leakage checks._

## Go / no-go
- _Decision and rationale before committing GPU time to pretraining._
