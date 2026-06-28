# Runbook — Running the Pipeline

End-to-end steps for what's built today. Stages not yet implemented are marked **TODO**. The
**primary corpus is Fannie Mae**; the Dutch synthetic panel is the validation set. See
`docs/architecture.md` for how the pieces fit and `CLAUDE.md` for project context.

## 0. Environment (H100 container)
```bash
bash scripts/setup_container.sh     # restart-proof venv + install + git (docs/container_setup.md)
```
Secrets (`WANDB_API_KEY`, `HF_TOKEN`) and the GCS key live in `/workspace/secrets.env` and
`/workspace/.gcloud/credit-fm-sa.json` — never committed. Storage is pluggable: every path may be
a local path or a `gs://` / `s3://` URL.

## 1. Data
- **Fannie Mae (primary):** ingested to GCS via the `fannie-mae-etl` repo →
  `scripts/ingest_fannie_mae.py` writes a per-loan monthly panel with derived `origination_date`,
  `reporting_date`, `default_event`, `is_performing`. See `docs/data/fannie_mae.md`.
- **Dutch (validation):** `data/raw/all_cutoffs.parquet` (ESMA Annex 2). `loan_book.parquet`
  carries eval-only latents (`_segment`) — **never a feature**.

## 2. Split (loan-stratified, temporal — DL-007)
```bash
python scripts/prepare_data.py --input <panel.parquet> --origination-col origination_date
# → processed/{train,val,test}.parquet + splits.csv + splits.meta.json
```
Schema-agnostic via `--id-col` / `--origination-col` (or derive with `--reporting-col`/`--seasoning-col`).

## 3. Tokenizer config (generated, not hand-edited)
```bash
python scripts/classify_schema.py --input processed/train.parquet \
  --out configs/<asset>/tokenizer.yaml
# → profile/event field roles; drops constant + redundant columns
```

## 4. Fit the tokenizer  ✅ M1
```bash
python scripts/train_tokenizer.py \
  --config configs/fannie_mae/tokenizer.yaml \
  --train  gs://sriram-credit-fm-data/output/processed/fannie_mae/run_2016_2017/train.parquet \
  --out    configs/fannie_mae/tokenizer.json \
  --report reports/fannie_tokenizer_report.md
# → frozen vocab (Fannie: 440 tokens) + QA report (roundtrip, OOV, length). Fit on TRAIN only.
```

## 5. Encode shards (encode-once)  ✅ M2
```bash
python scripts/encode_dataset.py \
  --tokenizer configs/fannie_mae/tokenizer.json \
  --in   gs://.../output/processed/fannie_mae/run_2016_2017/train.parquet \
  --out  gs://.../output/encoded/fannie_mae/run_2016_2017/train --shard-size 50000
# repeat for val/test → shard-*.parquet + manifest.json (token-id contract for the model)
```
In code, the data layer is then:
```python
from credit_fm.data import CreditDataModule
dm = CreditDataModule(train_dir, val_dir=val_dir, batch_size=64, limit=1000)  # limit = toy run
batch = next(iter(dm.train_dataloader()))   # {input_ids, attention_mask, labels, event_index, ...}
```

## 6. Pretrain the model (MLM)  ✅ M2
```bash
# single-GPU run (encode shards first, step 5). Watch TRAIN vs VAL loss.
python scripts/pretrain.py \
  --tokenizer configs/fannie_mae/tokenizer.json \
  --train-dir gs://.../output/encoded/fannie_mae/run_2016_2017/train \
  --val-dir   gs://.../output/encoded/fannie_mae/run_2016_2017/val \
  --limit 100000 --steps 1500 --batch-size 128 --dim 384 --bf16 --dropout 0.1 \
  --val-every 150 --log-every 50 --out gs://.../runs/toy.pt
```
**Read the loss:** train and val falling *together* = generalising; a wide gap = overfitting.
`train_mlm` early-stops on best val and saves those weights. At small data (≤100k loans) the model
overfits by design — see **DL-015**: the 25.5M model needs ~2M loans (~500M tokens). MLM loss is a
proxy; the real gate is the downstream OOT eval (§8). Scale `--dim`/`--steps`/`--limit` for M3.

## 7. Baselines — the honest bar for the FM
```bash
# Fannie out-of-time (OOT) — the real bar
python scripts/build_oot_baseline.py --train-years 2000-2006 --test-years 2008-2010 \
  --sample-pct 20 --report reports/fannie_oot_crisis.md     # crisis stress: ROC 0.757 / PR 0.024
python scripts/build_oot_baseline.py --train-years 2000-2022 --test-years 2023-2025 \
  --sample-pct 20 --report reports/fannie_oot_recent.md     # recent OOT (run via nohup)

# Dutch single-cutoff baseline + Gate G1 + hidden-segment ceiling
python scripts/train_baseline.py --config configs/dutch_mortgages/baseline.yaml \
  --book data/raw/loan_book.parquet --report reports/baseline_report.md   # Gate G1 = ROC 0.73
```

## Verify
```bash
ruff check . && pytest -q
```

## New asset class
Generic split + classify work as-is. Write `configs/<asset>/{baseline,tokenizer}.yaml`, then run
steps 2–5 — no code change.

## 8. Downstream eval — the real verdict (planned, Phase E)
The FM is judged by whether its `[USR]` embeddings beat the **OOT baseline (ROC 0.757)** at default
prediction — not by MLM loss (DL-015). To be built:
- `scripts/extract_embeddings.py` — `[USR]` embeddings from a checkpoint.
- `scripts/evaluate_downstream.py` — FM embeddings vs the OOT baseline (the FM-vs-0.757 test).

## Not built yet (TODO — land as stages complete)
- **Parallel encoding** (`encode_dataset.py --workers`) — needed for the full-corpus M3 run (DL-015).
- 8× H100 multi-GPU pretraining + W&B logging (M3; DL-009).
