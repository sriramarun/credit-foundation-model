# Tokenization

Key-value-time scheme. Each field → semantic-type (key) token + value token(s) + temporal
coordinate.

- **Keys**: ~70 tokens (one per field name).
- **Values**: 200–500 tokens (16–32 percentile buckets per continuous field, single token
  per categorical value, BPE subword for text).
- **Temporal**: `8*ln(1+seconds/8)` log-seconds since last event + cyclical
  (sin/cos) hour/day/week features, added to event-token embeddings.

Sequence layout: `[BOS]` + origination block + per-cutoff event blocks
(`[EVT_START]`…`[EVT_END]`) + `[EOS]`.

## Vocabulary fitting (leakage rule)

The vocabulary and all numeric bin edges are fit on **`data/processed/train.parquet` only**
(see Decision Log DL-008). Build the split first (`scripts/prepare_data.py`), then fit the
tokenizer on `train` — fitting on val/test/full leaks distribution into the tokenizer.

## Field classification & config generation

Per-asset field roles live in `configs/<asset>/tokenizer.yaml`, generated **reproducibly
from the data** by `scripts/classify_schema.py` — do not hand-edit (the file header records
the exact regenerate command). For each column it determines:

- **role** — `id` / `static` (constant within a loan → Profile State Encoder) / `dynamic`
  (varies per cutoff → Event Encoder).
- **type** — `numeric` (→ percentile buckets), `categorical` / `bucket` / `flag` (→ single
  token), `temporal`, or `constant`.

It then drops two groups (`find_redundant`):

- `drop_constant` — a single value across the panel (no signal).
- `drop_redundant` — auto-detected exact-duplicate columns and `*_bucket` discretizations of
  a numeric field, plus any functional-dependency candidates opted into via `--drop` (kept by
  default, so explicit signals like `default_crr_flag` are never lost silently).

**Dutch mortgages:** 71 columns → **42 features** (29 static, 13 dynamic); 11 constant + 15
redundant dropped (validated against the ESMA column glossary, 70/71 match — see Decision Log
DL-010). Regenerate from the train split:

    python scripts/classify_schema.py --input data/processed/train.parquet \
        --out configs/dutch_mortgages/tokenizer.yaml \
        --drop interest_only_flag,self_employed_flag,property_usage,buy_to_let_flag,days_past_due,primary_energy_demand_kwh_m2,construction_year_bucket
