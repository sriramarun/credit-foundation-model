# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Ingest Fannie Mae Single-Family Loan Performance quarterly parquet snapshots.

Source: ~100 quarterly parquet files in a GCS bucket (one per acquisition quarter,
e.g. ``2018Q1.parquet`` holds every monthly performance row for loans *acquired* that
quarter across their whole life). This script reads a chosen set of quarters, normalises
the column names to the canonical snake_case schema (``docs/data/fannie_mae.md``), derives
the modelling columns the generic pipeline needs, and writes:

  * ``<out>/quarter=<YYYYQn>/part.parquet``  — per-quarter, for scale-out later, and
  * ``<out>/panel.parquet``                  — the combined sample the dev pipeline reads.

Derived columns (so ``train_baseline.py`` / ``prepare_data.py`` stay asset-generic):
  * ``reporting_date``     month-end Timestamp parsed from Monthly Reporting Period (MMYYYY)
  * ``origination_date``   Timestamp parsed from Origination Date (MMYYYY) — real, not derived
  * ``dlq_num``            Current Loan Delinquency Status as int (``XX``/blank -> NaN)
  * ``default_event``      True if dlq_num >= 6 (D180) OR zero_balance_code in a credit event
  * ``prepay_event``       True if zero_balance_code == '01' (prepaid/matured)
  * ``is_performing``      True if current (dlq_num == 0) and not yet terminated

Dev sample first:
    python scripts/ingest_fannie_mae.py \
        --gcs gs://<bucket>/<prefix> --quarters 2018Q1 2018Q2 \
        --out data/raw/fannie_mae
Then scale by widening --quarters (or --all) once the pipeline is proven end to end.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Zero Balance Code -> outcome. Credit events count as default; 01 is a clean prepay.
ZBC_CREDIT_EVENT = {"02", "03", "09", "15"}  # third-party sale, short sale, REO/DIL, note sale
ZBC_PREPAY = {"01"}                          # prepaid or matured
D180 = 6                                     # months delinquent that defines a default

# Aliases -> canonical name, for the handful of columns the derivations depend on.
# Fannie's published parquet uses these names; map common variants defensively. After the
# first dev run the column report prints what was actually found, so this can be tightened.
CRITICAL_ALIASES = {
    "loan_identifier": "loan_id",
    "loan id": "loan_id",
    "monthly_reporting_period": "reporting_period",
    "monthly reporting period": "reporting_period",
    "origination_date": "orig_date",
    "origination date": "orig_date",
    "current_loan_delinquency_status": "current_dlq_status",
    "current loan delinquency status": "current_dlq_status",
    "loan_delinquency_status": "current_dlq_status",
    "zero_balance_code": "zero_balance_code",
    "zero balance code": "zero_balance_code",
}
REQUIRED = ["loan_id", "reporting_period", "orig_date", "current_dlq_status", "zero_balance_code"]


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: re.sub(r"\s+", "_", str(c).strip().lower()))
    return df.rename(columns={k.replace(" ", "_"): v for k, v in CRITICAL_ALIASES.items()})


def _parse_mmyyyy(s: pd.Series) -> pd.Series:
    """Fannie dates are MMYYYY (sometimes M/D/YYYY in older mirrors); coerce to month-end."""
    txt = s.astype("string").str.strip()
    dt = pd.to_datetime(txt, format="%m%Y", errors="coerce")
    if dt.isna().mean() > 0.5:                       # fall back to a permissive parse
        dt = pd.to_datetime(txt, errors="coerce")
    return dt + pd.offsets.MonthEnd(0)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    df["reporting_date"] = _parse_mmyyyy(df["reporting_period"])
    df["origination_date"] = _parse_mmyyyy(df["orig_date"])

    dlq = df["current_dlq_status"].astype("string").str.strip()
    df["dlq_num"] = pd.to_numeric(dlq, errors="coerce")  # 'XX'/'' -> NaN
    zbc = df["zero_balance_code"].astype("string").str.strip().str.zfill(2)

    df["default_event"] = (df["dlq_num"] >= D180) | zbc.isin(ZBC_CREDIT_EVENT)
    df["prepay_event"] = zbc.isin(ZBC_PREPAY)
    df["is_performing"] = (df["dlq_num"] == 0) & (~zbc.isin(ZBC_CREDIT_EVENT | ZBC_PREPAY))
    return df


def _column_report(df: pd.DataFrame) -> None:
    print("  columns:", len(df.columns))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"  !! MISSING canonical columns {missing} — add aliases to CRITICAL_ALIASES")
        print(f"     available: {sorted(df.columns)[:40]}{' ...' if len(df.columns) > 40 else ''}")


def _read_one(base: str, quarter: str) -> pd.DataFrame:
    # try a few common file naming conventions under the prefix
    for name in (f"{quarter}.parquet", f"{quarter}/part.parquet", f"{quarter}/*.parquet"):
        path = f"{base.rstrip('/')}/{name}"
        try:
            return pd.read_parquet(path)
        except Exception:
            continue
    raise SystemExit(f"Could not read quarter {quarter} under {base} (tried .parquet / dir forms)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gcs", help="gs://bucket/prefix holding the quarterly parquet files")
    src.add_argument("--local", help="local dir holding the quarterly parquet files")
    ap.add_argument("--quarters", nargs="+", help="e.g. 2018Q1 2018Q2 (dev sample)")
    ap.add_argument("--out", default="data/raw/fannie_mae")
    ap.add_argument("--combined-name", default="panel.parquet")
    args = ap.parse_args()

    base = args.gcs or args.local
    if not args.quarters:
        raise SystemExit("Pass --quarters (dev: a quarter or two). Full-scale ingest comes later.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for q in args.quarters:
        print(f"Reading {q} from {base} ...")
        df = _norm_cols(_read_one(base, q))
        _column_report(df)
        df = _derive(df)
        qdir = out / f"quarter={q}"
        qdir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(qdir / "part.parquet", index=False)
        print(f"  {q}: {len(df):>10,} rows  "
              f"default_rate={df['default_event'].mean():.4%}  "
              f"performing={df['is_performing'].mean():.1%}")
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(out / args.combined_name, index=False)
    print(f"\nWrote {out}/{args.combined_name}: {len(panel):,} rows, "
          f"{panel['loan_id'].nunique():,} loans, "
          f"reporting {panel['reporting_date'].min().date()} -> {panel['reporting_date'].max().date()}")
    print("Next: scripts/prepare_data.py --input "
          f"{out}/{args.combined_name} --origination-col origination_date")


if __name__ == "__main__":
    main()