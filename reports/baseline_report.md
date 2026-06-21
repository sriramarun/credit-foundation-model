# Baseline Report — XGBoost (Gate G1)

Config `configs/dutch_mortgages/baseline.yaml` · split `data/processed` (temporal, DL-007). Observation `2024-12-31`; label = `default_crr_flag`==Y within 6 cutoffs.

| Config | test ROC-AUC | test PR-AUC | pos% |
|---|--:|--:|--:|
| (1) full features, no gate | 0.9288 | 0.8045 | 4.00% |
| (2) full features + gate | 0.8382 | 0.4345 | 1.71% |
| (3) no-leakage features, no gate | 0.7356 | 0.0994 | 4.00% |
| (4) no-leakage + gate (Gate G1) | 0.7391 | 0.0463 | 1.71% |

## Reading it
- **Leakage** (contemporaneous state) inflates (1); removing it → (3).
- **Gate** = predict *new* events among currently-performing loans (realistic).
- **Gate G1 = config (4)**: ROC-AUC 0.739, PR-AUC 0.046.

## Caveat
- Synthetic data is rule-based, so the clean baseline runs high.

## Architectural validation — the hidden `_segment` ceiling

The generator assigns each loan a hidden fragility latent `_segment` (in `loan_book`, not the panel — evaluation-only, never a feature). It drives default but is invisible to tabular models. On the Gate-G1 cohort:

| (A) Segment | loans (test) | default rate |
|---|--:|--:|
| 0 | 28,768 | 0.28% |
| 1 | 11,411 | 1.37% |
| 2 | 5,679 | 9.60% |
| **spread** | | **34×** |

**(B) Oracle-`_segment` lift** — if the model could see the latent:

| | ROC-AUC | PR-AUC |
|---|--:|--:|
| Gate G1 (observables only) | 0.739 | 0.046 |
| + oracle `_segment` (diagnostic) | 0.844 | 0.084 |
| **headroom** | **+0.105** | **+82%** |

**(C) Can XGBoost recover `_segment`?** accuracy 65.2% vs 62.7% majority-class (macro-F1 0.42) — **essentially no.** The signal exists (B) but tabular observables can't reach it.

**Conclusion.** A real, large source of default risk (A) is invisible to point-in-time tabular models (C); recovering it would nearly double PR-AUC (B). The foundation model reads each loan's behavioural *sequence* to recover that latent — that headroom above the baseline is the project's thesis, now quantified.

Reproduce: `python scripts/train_baseline.py --config configs/dutch_mortgages/baseline.yaml --data-dir data/processed --book data/raw/loan_book.parquet --report reports/baseline_report.md`
