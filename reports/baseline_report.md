# Baseline Report — XGBoost (Gate G1)

Panel split: `data/processed` (loan-stratified temporal, DL-007). Observation `2024-12-31`; label = CRR default in the next 6 months.

Four configurations isolate signal from leakage; the honest bar the foundation model must beat is **config (4)**.

| Config | test ROC-AUC | test PR-AUC | pos% |
|---|--:|--:|--:|
| (1) full features, no gate | 0.9288 | 0.8045 | 4.00% |
| (2) full features + performing gate | 0.8382 | 0.4345 | 1.71% |
| (3) no-leakage features, no gate | 0.7356 | 0.0994 | 4.00% |
| (4) no-leakage + gate (Gate G1) | 0.7391 | 0.0463 | 1.71% |

## Reading it
- **Leakage** (contemporaneous delinquency state): config (1)→(3) drops ROC-AUC sharply — those features read the current state.
- **Performing-at-obs gate**: predict *new* defaults among performing loans → PR-AUC collapses at a low base rate (the realistic task).
- **Gate G1 = config (4)**: ROC-AUC 0.739, PR-AUC 0.046.

## Caveat
- Synthetic data is rule-based, so the clean baseline runs higher than a real portfolio would.

## Architectural validation — the hidden-segment ceiling

The generator assigns each loan a hidden fragility **segment** (in `loan_book`, not the ESMA panel — evaluation-only, never a feature). It drives default but is invisible to tabular models. Measured on the Gate-G1 cohort:

| (A) Segment | loans (test) | default rate |
|---|--:|--:|
| stable | 28,768 | 0.28% |
| baseline | 11,411 | 1.37% |
| fragile | 5,679 | 9.60% |
| **spread** | | **34×** |

**(B) Oracle-segment lift** — if the model could see the segment:

| | ROC-AUC | PR-AUC |
|---|--:|--:|
| Gate G1 (observables only) | 0.739 | 0.046 |
| + oracle `_segment` (diagnostic) | 0.844 | 0.084 |
| **headroom** | **+0.105** | **+82%** |

**(C) Can XGBoost recover the segment?** accuracy 65.2% vs 62.7% majority-class (macro-F1 0.42) — **essentially no.** The signal exists (B) but tabular observables can't reach it.

**Conclusion.** The hidden segment is a real, large source of default risk (A) that point-in-time tabular models cannot see (C); recovering it would nearly double PR-AUC (B). The foundation model reads each loan's behavioural *sequence* to recover that latent — that headroom above 0.73 is the project's thesis, now quantified.

Reproduce: `python scripts/train_baseline.py --data-dir data/processed --book data/raw/loan_book.parquet --report reports/baseline_report.md`
