# Extending to a New Asset Class

Adaptation is configuration, not code. A new asset class is a new `configs/<asset>/` directory
of YAML recipes run through the same scripts:

1. **Ingest** the raw source into the canonical per-loan monthly panel (one row per
   loan-month, an id column, an ISO reporting date, and derived label/gate columns). Write an
   asset ingest script only if the source needs bespoke parsing — everything after ingest is
   generic.
2. **Task schema** — `configs/<asset>/baseline.yaml`: id/time/label/gate roles plus the
   `exclude` and `leakage` column lists (the no-peeking contract every stage reads).
3. **Split** — `scripts/prepare_data.py -c configs/<asset>/prepare.yaml` (loan-stratified,
   temporal by origination; derives origination from `reporting − seasoning` if the panel has
   no origination column). Audit with `scripts/validate_splits.py`.
4. **Field schema** — `scripts/classify_schema.py -c configs/<asset>/classify.yaml` suggests
   the profile/event/type routing from the train split; review it (especially against your
   leakage list) into `configs/<asset>/tokenizer.yaml`.
5. **Standard pipeline** — the same recipes, one per stage: `train_tokenizer` → `encode_dataset`
   → `pretrain` → `extract_embeddings` → `evaluate_downstream` / `finetune`. All scripts share
   one grammar: `-c recipe.yaml --key.path override`.

The **Dutch mortgages** configs (`configs/dutch_mortgages/`) are the worked example: a
completely different schema (ESMA Annex 2, 71 columns, no origination-date column) running
through identical scripts — the delta vs the mortgage reference is YAML only.
