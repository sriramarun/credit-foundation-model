# Fannie Mae Single-Family Loan Performance — data notes

**Primary pretraining corpus.** Real-world US single-family **fixed-rate** mortgages, ~25 years,
sourced as ~100 **quarterly parquet snapshots** from a GCS bucket (one file per *acquisition
quarter*; each file holds every monthly performance row for the loans acquired that quarter
across their whole life). Source layout: Fannie Mae *Single-Family Loan Performance Dataset and
Credit Risk Transfer — Glossary and File Layout* (113 fields).

## Why it's the primary set (vs the Dutch synthetic panel)
- **Real data** — makes the "FM beats tabular" thesis credible, not a synthetic artefact.
- **True `Origination Date`** (field 14) — temporal split orders by it directly
  (`prepare_data.py --origination-col origination_date`); no seasoning-derivation (DL-007).
- **Already a monthly performance panel** (field 3, Monthly Reporting Period) — maps onto the
  event/history branches with no reshaping.
- The Dutch synthetic panel stays as a **controlled validation/ablation set** — it carries the
  hidden `_segment` latent that gives us the segment-ceiling proof (not possible on real data).

## Structure
- **Grain:** one row per `loan_id` per `reporting_date` (monthly).
- **Static / profile fields** (constant per loan): channel, seller, original interest rate,
  original UPB, original loan term, origination date, original LTV / CLTV, number of borrowers,
  DTI, FICO at origination, first-time-homebuyer, loan purpose, property type, number of units,
  occupancy, property state, MI %, amortization type, etc.
- **Dynamic / event fields** (per period): current interest rate, current actual UPB, loan age,
  remaining months to maturity, current FICO.

## Label (derived in `scripts/ingest_fannie_mae.py`)
- **`default_event`** = `Current Loan Delinquency Status` reaches **D180** (≥ 6 months delinquent)
  **OR** `Zero Balance Code` ∈ {02 third-party sale, 03 short sale, 09 REO/deed-in-lieu,
  15 note sale} — the standard Fannie credit-event definition.
- **`prepay_event`** = `Zero Balance Code` == 01 (prepaid / matured).
- **`is_performing`** = current (dlq 0) and not yet terminated — the performing-at-observation gate.

## Leakage discipline (same rules as Dutch — see `configs/fannie_mae/baseline.yaml`)
Contemporaneous-state / outcome-revealing fields are dropped for default prediction and the task
is gated to performing-at-observation: current delinquency status, zero-balance code + dates,
foreclosure/disposition dates, last-paid-installment, and all loss / proceeds / expense / deferral
/ modification-loss columns. Split by `loan_id` (never row); temporal by origination; vocab/bins
fit on `train` only.

## Ingest → pipeline (dev sample first)
```bash
# 1. sample a quarter or two from GCS -> parquet (+ derived columns)
python scripts/ingest_fannie_mae.py --gcs gs://<bucket>/<prefix> \
    --quarters 2018Q1 2018Q2 --out data/raw/fannie_mae
# 2. loan-stratified temporal split on the REAL origination date
python scripts/prepare_data.py --input data/raw/fannie_mae/panel.parquet \
    --origination-col origination_date
# 3. generate the tokenizer config from the schema
python scripts/classify_schema.py --input data/raw/fannie_mae/panel.parquet \
    --out configs/fannie_mae/tokenizer.yaml
# 4. honest XGBoost baseline (Gate G1 for Fannie)
python scripts/train_baseline.py --config configs/fannie_mae/baseline.yaml \
    --report reports/fannie_baseline_report.md
```
Then scale by widening `--quarters` once the end-to-end run is green.

## Open items
- Confirm exact column names in the GCS parquet (the ingest column report prints what it finds;
  tighten `CRITICAL_ALIASES` / the `exclude`/`leakage` lists to match).
- Align `obs_date` in `baseline.yaml` to the ingested sample's reporting range.
- GCS auth on the container: service-account JSON via `GOOGLE_APPLICATION_CREDENTIALS`
  (in `/workspace/secrets.env`, never committed); `gcsfs` installed.