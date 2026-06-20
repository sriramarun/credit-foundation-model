# Baseline Report — XGBoost (Gate G1)

Panel split: `data/processed` (loan-stratified temporal, DL-007). Observation `2024-12-31`; label = CRR default in the next 6 months.

Four configurations isolate signal from leakage. The honest baseline the foundation model must beat is **config (4)**.

| Config | test ROC-AUC | test PR-AUC | pos% |
|--------|----:|----:|----:|
| (1) full features, no gate | 0.9288 | 0.8045 | 4.00% |
| (2) full features + performing gate | 0.8382 | 0.4345 | 1.71% |
| (3) no-leakage features, no gate | 0.7356 | 0.0994 | 4.00% |
| (4) no-leakage + gate (Gate G1) | 0.7391 | 0.0463 | 1.71% |

## Reading it
- **Leakage** (contemporaneous `arrears_bucket`/`performing_status`/`default_crr_flag`/…): config (1)→(3) drops ROC-AUC sharply — those features read the current delinquency state.
- **Performing-at-obs gate**: predict *new* defaults among currently-performing loans → PR-AUC collapses at a low base rate (the realistic, hard task).
- **Gate G1 = config (4)**: ROC-AUC 0.739, PR-AUC 0.046.

## Caveats
- Synthetic data is rule-based, so even the clean baseline is higher than a real portfolio would give.
- Architectural-validation (segment latent ceiling) requires `loan_book.parquet` with `_segment`/`_latent_fragility`.

Reproduce: `python scripts/train_baseline.py --data-dir data/processed --report reports/baseline_report.md`
