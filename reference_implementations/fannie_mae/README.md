# Reference implementation — Fannie Mae Single-Family (PRIMARY)

Real-world US single-family fixed-rate mortgages (~25 years), the primary corpus for pretraining
the credit foundation model. See [`docs/data/fannie_mae.md`](../../docs/data/fannie_mae.md) for
the schema, label (D180 / Zero-Balance credit event), and leakage discipline.

## Pipeline
| Step | Command | Output |
|------|---------|--------|
| Ingest (dev sample) | `python scripts/ingest_fannie_mae.py --gcs gs://<bucket>/<prefix> --quarters 2018Q1 2018Q2 --out data/raw/fannie_mae` | `data/raw/fannie_mae/panel.parquet` |
| Split | `python scripts/prepare_data.py --input data/raw/fannie_mae/panel.parquet --origination-col origination_date` | `data/processed/{train,val,test}.parquet` |
| Tokenizer config | `python scripts/classify_schema.py --input data/raw/fannie_mae/panel.parquet --out configs/fannie_mae/tokenizer.yaml` | `configs/fannie_mae/tokenizer.yaml` |
| Baseline (Gate G1) | `python scripts/train_baseline.py --config configs/fannie_mae/baseline.yaml --report reports/fannie_baseline_report.md` | `reports/fannie_baseline_report.md` |

## Config
- [`configs/fannie_mae/baseline.yaml`](../../configs/fannie_mae/baseline.yaml) — label / gate / leakage map.
- `configs/fannie_mae/tokenizer.yaml` — **generated** by `classify_schema.py` (not hand-edited).

## Status
Scaffolding in place; dev-sample ingest pending GCS access on the container. Full-scale ingest
(all ~100 quarters) follows once the end-to-end dev run is green.
