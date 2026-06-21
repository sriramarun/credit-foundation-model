# Runbook — Running the Pipeline

How to run what's built today, end to end. Stubbed stages (pretrain, embeddings, downstream)
are marked TODO and will be added as they land. See `CLAUDE.md` for project context.

## 0. Environment (H100 container)
```bash
bash scripts/setup_container.sh          # restart-proof venv + install + git; see docs/container_setup.md
Secrets (WANDB_API_KEY, HF_TOKEN) in /workspace/secrets.env (never committed).

1. Data → data/raw/ (gitignored)
all_cutoffs.parquet — the ESMA panel (HF Algoritmica/green-lion-2024-2025 / deeploans generator).
loan_book.parquet — eval-only latents (_segment); use the generator run whose _segment
predicts this panel's defaults (verify via a segment-conditional default rate).
2. Split (generic)
python scripts/prepare_data.py --input data/raw/all_cutoffs.parquet
# → data/processed/{train,val,test}.parquet + splits.csv + splits.meta.json
Schema-agnostic: --id-col, --origination-col (or derive via --reporting-col/--seasoning-col).

3. Tokenizer config (generic)
python scripts/classify_schema.py --input data/processed/train.parquet \
  --out configs/dutch_mortgages/tokenizer.yaml \
  --drop interest_only_flag,self_employed_flag,property_usage,buy_to_let_flag,days_past_due,primary_energy_demand_kwh_m2,construction_year_bucket
# → 42-feature config (drop_constant + drop_redundant + profile/event)
4. Baseline + Gate G1 + ceiling (config-driven)
python scripts/train_baseline.py --config configs/dutch_mortgages/baseline.yaml \
  --book data/raw/loan_book.parquet --report reports/baseline_report.md
# → 4-config table (Gate G1 = ROC 0.73 / PR-AUC 0.046) + hidden-segment ceiling (34×, +95%)
Verify
ruff check . && pytest -q
# optional: run notebooks/00_smoke_test_splits.ipynb
New asset class
Generic split + classify work as-is. Write configs/<asset>/baseline.yaml
(id/time/label/horizon/gate/leakage/segment) — no code change to train_baseline.py.

Not built yet (TODO — added as stages land)
scripts/train_tokenizer.py — fit the KVT tokenizer (Milestone M1)
scripts/pretrain.py — pretrain the 3-branch model on 8× H100 (M3)
scripts/extract_embeddings.py — embeddings from a checkpoint
scripts/evaluate_downstream.py — embeddings vs baseline (the FM-vs-0.73 test)