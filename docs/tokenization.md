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
