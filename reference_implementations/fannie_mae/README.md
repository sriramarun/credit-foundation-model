# Reference implementation — Fannie Mae Single-Family (PRIMARY)

Real-world US single-family fixed-rate mortgages (~25 years), the primary corpus for pretraining
the credit foundation model. See [`docs/data/fannie_mae.md`](../../docs/data/fannie_mae.md) for
the schema, label (D180 / Zero-Balance credit event), and leakage discipline.

## Pipeline
| Step | Command | Output |
|------|---------|--------|
| Ingest (GCS, multi-year span) | `python scripts/ingest_fannie_mae.py --gcs-root gs://sriram-credit-fm-data/fannie_by_reporting --reporting 2016Q1 2016Q2 … 2017Q4` | `data/raw/fannie_mae/panel.parquet` |
| Split (persisted to bucket) | `python scripts/prepare_data.py --input data/raw/fannie_mae/panel.parquet --origination-col origination_date --out-dir gs://sriram-credit-fm-data/processed/fannie_mae/run_2016_2017` | `<out-dir>/{train,val,test}.parquet` + `splits.{csv,meta.json}` |
| Tokenizer config | `python scripts/classify_schema.py --input data/raw/fannie_mae/panel.parquet --out configs/fannie_mae/tokenizer.yaml` | `configs/fannie_mae/tokenizer.yaml` |
| Baseline (Gate G1) | `python scripts/train_baseline.py --config configs/fannie_mae/baseline.yaml --data-dir gs://sriram-credit-fm-data/processed/fannie_mae/run_2016_2017 --report reports/fannie_baseline_report.md` | `reports/fannie_baseline_report.md` |

`--out-dir`/`--data-dir` are pluggable (`credit_fm.utils.storage`): local, `gs://`, or `s3://`. Dev plumbing on explicit local files: `python scripts/ingest_fannie_mae.py --files data/raw/<f1>.parquet data/raw/<f2>.parquet`. A real Gate-G1 number needs a reporting span covering `obs_date + horizon` (one quarter only exercises the plumbing — the gated task has too few new defaults to score).

## Config
- [`configs/fannie_mae/baseline.yaml`](../../configs/fannie_mae/baseline.yaml) — label / gate / leakage map.
- `configs/fannie_mae/tokenizer.yaml` — **generated** by `classify_schema.py` (not hand-edited).

## Status
Ingest + split + baseline validated end-to-end on real sample files (schema, leakage lists, and
the pluggable GCS/S3 storage path all confirmed). Pending: run a multi-year span from GCS on the
container for the real Gate-G1, then scale to the full reporting history.