# Reference implementation — Fannie Mae Single-Family (mortgages)

The framework's reference corpus: ~25 years of real-world US fixed-rate mortgages (public
Fannie Mae Single-Family Loan Performance data). See
[`docs/data/fannie_mae.md`](../../docs/data/fannie_mae.md) for the schema, label
(D180 / Zero-Balance credit event), and leakage discipline, and
[`notebooks/00_data_bible.ipynb`](../../notebooks/00_data_bible.ipynb) for the full
column-by-column reference.

## Pipeline (config-driven — `-c recipe.yaml` + dotted overrides)

| Step | Command | Output |
|------|---------|--------|
| Ingest | `python scripts/ingest_fannie_mae.py -c configs/fannie_mae/ingest_2000_2024.yaml` | per-loan monthly panel (labels derived, 4% loan-hash sample) |
| Validate | `python scripts/validate_ingest.py --panel <out>/panel_2000_2024.parquet` | PASS/FAIL audit |
| Split | `python scripts/prepare_data.py -c configs/fannie_mae/prepare.yaml --run_name run_2000_2024 --reporting_max 2022-12-31` | `{train,val,test}.parquet` + `splits.{csv,meta.json}` |
| Validate | `python scripts/validate_splits.py --dir <out_dir>` | PASS/FAIL audit |
| Tokenizer fit | `python scripts/train_tokenizer.py -c configs/fannie_mae/tokenizer_fit.yaml` | `tokenizer.json` (552-token frozen vocab) + QA report |
| Encode | `python scripts/encode_dataset.py -c configs/fannie_mae/encode.yaml --workers 32` | token-id shards + manifest |
| Pretrain | `python scripts/pretrain.py -c configs/fannie_mae/pretrain.yaml` | MLM checkpoint |
| OOT verdict | `scripts/build_oot_baseline.py` + `extract_embeddings` + `finetune -c configs/fannie_mae/finetune_oot.yaml` | reports |

Every path is pluggable (`credit_fm.utils.storage`): local, `gs://`, or `s3://`.

## Config

- [`configs/fannie_mae/baseline.yaml`](../../configs/fannie_mae/baseline.yaml) — id/time/label/gate
  roles + the `exclude`/`leakage` lists (the no-peeking contract).
- `configs/fannie_mae/tokenizer.yaml` — field schema (profile/event routing, bins, anchors);
  curated on top of `classify_schema.py`'s suggestions (leakage exclusion, ARM/IO drops).
- `configs/fannie_mae/common.yaml` + one recipe per stage.

## Out-of-time evaluation — the bar and the result

`scripts/build_oot_baseline.py` builds the calendar-split bar: observe each loan every December
it is performing, label = default within 12 months, train on past years / test on future years,
with **loan-disjoint** and **embargo** guards.

| Experiment | Window | Result |
|---|---|---|
| XGBoost bar | train 2016–2021 → test 2022–2023 | ROC 0.7913 / AP 0.0057 |
| **FM full fine-tune** | identical window | **ROC 0.8257 / AP 0.0113** |
| Crisis stress (baseline) | train 2000–2006 → test 2008–2010 | ROC 0.757 / AP 0.024 |

Long runs: detach with `nohup … > logs/run.log 2>&1 &` and `tail -f` the log.
