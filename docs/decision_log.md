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
| DL-010 | Field selection: drop 11 constant + 16 redundant/derived | locked |

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

## DL-010 — Field selection (validated against the column glossary)
Cross-checked the empirical static/dynamic classification against the ESMA column glossary:
**70/71 match.** Confirmed the data-driven calls that `current_interest_rate_pct`,
`forbearance_flag`, `restructuring_flag` are static in this synthetic panel.

Final feature set = **42** (29 static → Profile encoder, 13 dynamic → Event encoder). Dropped:
- **11 constant** columns (single value across the panel — no signal).
- **16 redundant/derived** columns, each a deterministic function of a kept field: the
  exact duplicate `fixed_interest_period_end_in_months`; derived flags
  (`interest_only_flag`, `self_employed_flag`, `buy_to_let_flag`, `property_usage`);
  `epc_issue_year`, `primary_energy_demand_kwh_m2`, `days_past_due`, `maturity_date_proxy`;
  and the 7 pre-computed `*_bucket` columns (keep the raw continuous field and let the
  tokenizer percentile-bucket it).

**Data anomaly (flag to data team):** `guarantee_type` is spec'd as `{NHG, None}` but the file
has 1 value + 62.8% null; it equals `nhg_flag` and is dropped (no info lost).
