# Fannie Mae — FM Downstream Eval (loan-holdout probe)

Observed at **2016-12-31**, label = default within **12 months** after. 1,042,628 performing loans; loan-disjoint 30% held out. Features = tokenizer profile+event fields as-of cutoff (same info the FM saw).

| model | ROC-AUC | PR-AUC | dROC vs features |
|---|--:|--:|--:|
| features (XGB) | 0.8530 | 0.0142 | +0.0000 |
| FM embeddings (XGB) | 0.7992 | 0.0063 | -0.0538 |
| combined (XGB) | 0.8385 | 0.0090 | -0.0145 |
| FM embeddings (linear probe) | 0.8244 | 0.0091 | -0.0285 |

## Read
- If **FM embeddings** or **combined** beats **features**, the FM adds signal beyond the raw as-of-cutoff features — the thesis, on real loans.
- This is a loan-holdout probe on one window, not the calendar-OOT vs 0.757/0.784 (which needs a multi-year panel + train-years/test-years split).