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
## Out-of-time (OOT) baseline — the bar the foundation model must beat

`scripts/build_oot_baseline.py` builds a **calendar-split** baseline from the full history:
observe each loan every December it is performing, label = default within 12 months, then **train
on past years / test on future years**, with two guards — *loan-disjoint* (a loan is wholly in
train or test) and *embargo* (train label windows can't reach the test period). Reads the raw
acquisition-cohort files (`gs://…/parquet/<acqQ>.parquet`) so each loan's full life is co-located.
Configs: `configs/fannie_mae/baseline.yaml` (leakage/gate) + `configs/fannie_mae/raw_schema.yaml`.

| Experiment | Command | Result |
|---|---|---|
| Crisis stress test | `python scripts/build_oot_baseline.py --train-years 2000-2006 --test-years 2008-2010 --sample-pct 20 --report reports/fannie_oot_crisis.md` | ROC 0.757 / PR-AUC 0.024 |
| Recent OOT | `python scripts/build_oot_baseline.py --train-years 2000-2022 --test-years 2023-2025 --sample-pct 20 --report reports/fannie_oot_recent.md` | see report |

**Long runs (the full 26-year span) — run detached with `nohup`:**

```bash
mkdir -p logs
nohup python scripts/build_oot_baseline.py --train-years 2000-2022 --test-years 2023-2025 \
  --sample-pct 20 --report reports/fannie_oot_recent.md > logs/oot_recent.log 2>&1 &
echo $! > logs/oot_recent.pid       # PID, in case you need to check/stop it
tail -f logs/oot_recent.log         # follow progress; Ctrl-C stops watching, NOT the job
```
