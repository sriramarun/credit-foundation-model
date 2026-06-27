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
| DL-011 | Per-event calendar/macro-regime token `cal=<YYYYQ#>` | locked |
| DL-012 | Threshold-anchored + per-field numeric bins | locked |
| DL-013 | Hierarchical three-branch realization; architecture frozen at M2 | locked |
| DL-014 | Encode-once shards + flat `(B,L)` batches + 3-source MLM masking | locked |

## DL-007 — Loan-stratified temporal split
**Decision.** Split by `loan_id` (every cutoff of a loan stays in one split), ordered by
**origination date**, 80/10/10, train < val < test in time.

**Why.** Row splitting leaks loans (same loan, ±1 month, in train and test → fake test score).
Origination-ordered splitting mirrors production: a model trained on older loans is tested on
newer ones. Label-horizon leakage is handled at the label-generator layer.

**Artifacts.** `scripts/prepare_data.py` writes `data/processed/{train,val,test}.parquet` +
`splits.csv` + `splits.meta.json` (seed, source SHA-256, counts, ranges, git commit).

## DL-008 — Vocab on train only
Tokenizer vocabulary and numeric bin edges are fit on `train.parquet` only; never on
val/test/full panel (else test distribution leaks into the tokenizer).

## Note — data-module gitignore bug (2026-06-20)
`src/credit_fm/data/` (7 files) was never committed: the old unanchored `.gitignore` rule
`data/` matched `src/credit_fm/data/`. Restored; `.gitignore` is now anchored (`data/*`).

## DL-010 — Field selection (validated against the column glossary)
Final feature set = **42** (29 static → Profile, 13 dynamic → Event). Dropped **11 constant**
columns (no signal) and **16 redundant/derived** columns (deterministic functions of a kept
field, incl. the 7 pre-computed `*_bucket` columns — keep the raw field and let the tokenizer
bucket it).

## DL-011 — Calendar / macro-regime token
Each event block carries an absolute-time token `cal=<YYYYQ#>` (config `calendar: yearquarter`)
alongside the relative `t=<loan_age bin>`. Loan-internal tokens are regime-blind: 2005 and 2008
look identical, yet default behaviour is dominated by the macro cycle. The calendar token lets the
History encoder condition on the era; real macro series (HPI / rate / unemployment) will later
enter as ordinary `event` fields. **Fairness:** give the OOT baseline the same calendar/macro
features, or an FM win isn't apples-to-apples. (Reviewer #2.)

## DL-012 — Threshold-anchored + per-field numeric bins
Numeric fields default to 16 quantile bins, with two overrides: `bins:` raises resolution for
high-signal fields (LTV/CLTV/DTI/FICO/rate → 24), and `anchors:` forces a bin boundary exactly at
regulatory cliffs (LTV 80/90/95/97, DTI 36/43/45) so a hard underwriting threshold is never
blurred inside one bucket. Edges are still fit on `train` only (DL-008). (Reviewer #1.)

## DL-013 — Hierarchical realization; architecture frozen at M2
The three branches (DL-002) are realized **hierarchically**: the Event encoder contextualizes a
month's field tokens into one per-event vector; the History encoder attends across those event
vectors (length = #months, not #tokens) plus the profile vector; `[USR]` pools the loan
embedding. The full architecture is built and debugged at **toy scale in M2, then frozen** — M3
changes only data volume / compute, never architecture. This separates the "does it train?" risk
(M2) from the "does it scale/converge?" risk (M3). Reviewer #3 (downstream class imbalance) and #4
(length-bucketed batching) are deferred to Phases E and D.

## DL-014 — Encode-once shards + flat batches + 3-source masking
**Encode once.** The panel is tokenized a single time into token-id **shards** (one row per loan)
with four aligned arrays — `input_ids`, `event_index`, `field_type`, `branch` — so the DataLoader
never re-tokenizes (it would otherwise dominate GPU time). A `manifest.json` records tokenizer
version + source for reproducibility.

**Flat `(B, L)` batches.** Despite the hierarchy, batches are flat token sequences plus
`event_index`; the Event encoder pools per month from that index rather than using a nested
`(B, events, tokens)` axis. Less padding (loans vary widely in length), and the contract already
carries the indices. A varlen/packed variant (`PackedCollator`) is deferred to M3 for throughput.

**3-source MLM masking.** 15% token / 10% whole-event / 10% whole-field-type, BERT 80/10/10
corruption, specials never masked. Dynamic per batch for train; deterministic (seeded) for
val/test so loss is comparable across epochs. The three strategies each exercise a different
branch.
