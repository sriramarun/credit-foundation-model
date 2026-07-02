# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Ingest Fannie Mae Single-Family Loan Performance parquet snapshots.

Source (GCS bucket ``sriram-credit-fm-data``) is **Hive-partitioned by reporting period**:

    fannie_by_reporting/reporting_year=<YYYY>/reporting_quarter=<Q#>/from_<acqQ>_*.parquet

i.e. partitioned by the *observation* quarter; within each partition, files are sharded by
acquisition cohort (``from_2000Q1`` = loans originated ~2000Q1). The published schema is the
113-field Fannie layout with snake_case names (``loan_identifier``, ``monthly_reporting_period``,
``origination_date``, ``current_loan_delinquency_status``, ``zero_balance_code``, ...).

Reads a chosen set of slices (in parallel), derives the modelling columns the generic pipeline
needs, optionally loan-samples, and writes ``<out>/panel.parquet`` (local / gs:// / s3://).

Derived columns (so ``prepare_data.py`` / ``train_baseline.py`` stay asset-generic):
  * ``loan_id``           renamed from ``loan_identifier``
  * ``reporting_date``    ISO 'YYYY-MM-DD' month-end string from ``monthly_reporting_period`` (MMYYYY)
  * ``origination_date``  the MMYYYY string parsed in place to an ISO date string (the split key)
  * ``dlq_num``           ``current_loan_delinquency_status`` as int (``XX``/blank -> NaN)
  * ``default_event``     True if dlq_num >= 6 (D180) OR zero_balance_code is a credit event
  * ``prepay_event``      True if zero_balance_code == '01' (prepaid/matured)
  * ``is_performing``     True if current (dlq_num == 0) and not yet terminated

Config-driven (recipe: ``configs/fannie_mae/ingest.yaml``)::

    python -u scripts/ingest_fannie_mae.py -c configs/fannie_mae/ingest.yaml
    python -u scripts/ingest_fannie_mae.py -c configs/fannie_mae/ingest.yaml \
        --sources.reporting '[2016Q1, 2016Q2]' --sample_pct 100
    python -u scripts/ingest_fannie_mae.py -c configs/fannie_mae/ingest.yaml \
        --sources.files '[gs://.../from_2014Q1_0.parquet]'
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from credit_fm.utils import storage
from credit_fm.utils.config import parse_cli, summarize

# Zero Balance Code -> outcome. Credit events count as default; 01 is a clean prepay.
ZBC_CREDIT_EVENT = {"02", "03", "09", "15"}  # third-party sale, short sale, REO/DIL, note sale
ZBC_PREPAY = {"01"}                          # prepaid or matured
D180 = 6                                     # months delinquent that defines a default

# Real published column names the derivations depend on.
COL_ID = "loan_identifier"
COL_REPORTING = "monthly_reporting_period"
COL_ORIG = "origination_date"
COL_DLQ = "current_loan_delinquency_status"
COL_ZBC = "zero_balance_code"
REQUIRED = [COL_ID, COL_REPORTING, COL_ORIG, COL_DLQ, COL_ZBC]


def _iso_month_end(s: pd.Series) -> pd.Series:
    """Fannie dates are MMYYYY strings (e.g. '012016'); return ISO 'YYYY-MM-DD' month-end strings.

    The pipeline convention is an ISO-date *string* time column (chronologically sortable, and
    what ``train_baseline`` / ``prepare_data`` compare against) — not a timestamp.
    """
    txt = s.astype("string").str.strip()
    dt = pd.to_datetime(txt, format="%m%Y", errors="coerce")
    if dt.notna().any() and dt.isna().mean() > 0.5:   # permissive fallback for odd mirrors
        dt = pd.to_datetime(txt, errors="coerce")
    dt = dt + pd.offsets.MonthEnd(0)
    return dt.dt.strftime("%Y-%m-%d").where(dt.notna(), pd.NA)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing expected Fannie columns {missing}. Got {len(df.columns)} cols; "
            f"first 30: {sorted(df.columns)[:30]}")

    df = df.rename(columns={COL_ID: "loan_id"})
    df["reporting_date"] = _iso_month_end(df[COL_REPORTING])       # ISO 'YYYY-MM-DD' string
    df["origination_date"] = _iso_month_end(df[COL_ORIG])          # ISO string, in place

    dlq = df[COL_DLQ].astype("string").str.strip()
    df["dlq_num"] = pd.to_numeric(dlq, errors="coerce")           # 'XX'/'' -> NaN
    zbc = df[COL_ZBC].astype("string").str.strip().str.zfill(2)

    df["default_event"] = (df["dlq_num"] >= D180) | zbc.isin(ZBC_CREDIT_EVENT)
    df["prepay_event"] = zbc.isin(ZBC_PREPAY)
    df["is_performing"] = (df["dlq_num"] == 0) & (~zbc.isin(ZBC_CREDIT_EVENT | ZBC_PREPAY))
    return df


def _hive_path(root: str, reporting: str) -> str:
    """'2016Q1' -> '<root>/reporting_year=2016/reporting_quarter=Q1'."""
    year, _, q = reporting.partition("Q")
    if not (year.isdigit() and q.isdigit()):
        raise SystemExit(f"--reporting expects YYYYQn (e.g. 2016Q1), got '{reporting}'")
    return f"{root.rstrip('/')}/reporting_year={year}/reporting_quarter=Q{q}"


def _resolve_sources(cfg) -> list[str]:
    files = cfg.get_path("sources.files")
    if files:
        return list(files)
    root, reporting = cfg.get_path("sources.root"), cfg.get_path("sources.reporting")
    if not (root and reporting):
        raise SystemExit("config needs sources.files OR sources.root + sources.reporting "
                         "(e.g. [2016Q1, 2016Q2]).")
    return [_hive_path(root, r) for r in reporting]


def _maybe_auth(key: str) -> None:
    """Point gcsfs at the service-account key if one is available and not already set."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    if key and Path(key).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key
        print(f"Using GCS key: {key}")


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/fannie_mae/ingest.yaml")
    print(f"config: {cfg.config_path}\n"
          f"{summarize(cfg, 'sources', 'out', 'sample_pct', 'workers')}", flush=True)

    _maybe_auth(cfg.key)
    sources = _resolve_sources(cfg)
    out = cfg.out.rstrip("/")                           # local path or gs:///s3:// URL
    storage.ensure_auth(out, cfg.key)

    def _read_source(s: str) -> pd.DataFrame:
        df = _derive(storage.read_parquet(s))          # fsspec read: file or hive dir, local/gs://
        if cfg.sample_pct < 100:
            keep = pd.util.hash_pandas_object(df["loan_id"], index=False) % 100 < cfg.sample_pct
            df = df[keep]
        print(f"  done {s}: {len(df):>10,} rows  {df['loan_id'].nunique():>8,} loans  "
              f"reporting {df['reporting_date'].min()}..{df['reporting_date'].max()}", flush=True)
        return df

    print(f"Reading {len(sources)} source(s) with {cfg.workers} parallel workers ...", flush=True)
    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        frames = list(ex.map(_read_source, sources))   # order preserved
    panel = pd.concat(frames, ignore_index=True)
    panel_path = storage.join(out, cfg.combined_name)
    storage.write_parquet(panel, panel_path)            # pluggable: local / gs:// / s3://
    print(f"\nWrote {panel_path}: {len(panel):,} rows, "
          f"{panel['loan_id'].nunique():,} loans, "
          f"reporting {panel['reporting_date'].min()} -> {panel['reporting_date'].max()}, "
          f"origination {panel['origination_date'].min()} -> {panel['origination_date'].max()}")
    print("Next: python scripts/prepare_data.py -c configs/fannie_mae/prepare.yaml")


if __name__ == "__main__":
    main()
