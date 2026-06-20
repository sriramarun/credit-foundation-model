# Decision Log

Running record of locked decisions. Each entry: decision, rationale, status.

| ID | Decision | Status |
|----|----------|--------|
| DL-001 | Encoder-only + MLM (not decoder/causal) | locked |
| DL-002 | Three-branch encoders (Profile / Event / History) | locked |
| DL-003 | Key-value-time disentangled tokenization | locked |
| DL-004 | 30M default model size (Chinchilla-honest on ~600M tokens) | locked |
| DL-005 | Apache 2.0 | locked |
| DL-006 | HuggingFace primary, NeMo optional | locked |
| DL-007 | Loan-stratified temporal split; derived origination | locked |
| DL-008 | Tokenizer vocab + numeric bins fit on `train` only | locked |
| DL-009 | W&B hosted vs offline/self-hosted | **open** (resolve before pretraining) |

## DL-007 — Loan-stratified temporal split
**Decision.** Split by `loan_id` (every cutoff of a loan stays in one split), ordered by
**origination date**, 80/10/10, train < val < test in time.

**Origination key.** The ESMA panel has no origination-date column (`closing_date` is a
constant pool date, `reporting_date` is the cutoff). We derive a month-precise origination
= `reporting_date - seasoning_months` — verified 100% constant per loan and 100% consistent
with `origination_year`; range 2008-07 → 2023-07.

**Why.** Row splitting leaks loans (same loan, ±1 month, in train and test → fake test
score). Origination-ordered splitting mirrors production: a model trained on older loans is
tested on newer ones. Label-horizon leakage (e.g. `default_within_6m` needing the cutoff
≥6 months before panel end) is handled at the label-generator layer, not here.

**Artifacts.** `scripts/prepare_data.py` writes `data/processed/{train,val,test}.parquet` +
`splits.csv` + `splits.meta.json` (seed, source SHA-256, loan counts, origination ranges,
git commit) as the reproducibility/audit trail.

## DL-008 — Vocab on train only
Tokenizer vocabulary and numeric bin edges are fit on `train.parquet` only; never on
val/test/full panel (else test distribution leaks into the tokenizer).

## Note — data-module gitignore bug (2026-06-20)
`src/credit_fm/data/` (7 files) was never committed: the old unanchored `.gitignore` rule
`data/` matched `src/credit_fm/data/`. Restored; `.gitignore` is now anchored (`data/*`).
