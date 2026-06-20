# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Split a raw credit panel into loan-stratified temporal train/val/test parquets.

Writes ``data/processed/{train,val,test}.parquet`` (the whole 24-cutoff history of a loan
stays in one split), a ``splits.csv`` (``loan_id -> split``), and ``splits.meta.json`` — a
reproducibility/audit trail (seed, source SHA-256, loan counts, origination ranges, commit).

Origination key (what the temporal split orders by) comes from one of two modes:
  * ``--origination-col COL``  — use an explicit origination-date column directly.
  * derive (default)           — the Dutch RMBS panel has no origination-date column, so
    derive a month-precise origination from ``reporting_date - seasoning_months``
    (verified constant per loan and consistent with ``origination_year``).

Example (Dutch mortgages, derive mode):
    python scripts/prepare_data.py --input data/raw/Overall_2024_2025_all_months.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

from credit_fm.data.splits import SPLITS, temporal_loan_split
from credit_fm.utils.reproducibility import set_seed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _loan_origination(panel: pd.DataFrame, args) -> pd.Series:
    """Return one origination date per loan, indexed by id_col."""
    if args.origination_col:
        if args.origination_col not in panel.columns:
            raise SystemExit(
                f"Column '{args.origination_col}' not in panel. Available: {list(panel.columns)}")
        s = panel.groupby(args.id_col)[args.origination_col].min()
        return pd.to_datetime(s)

    # derive month-precise origination = reporting_date - seasoning_months
    for col in (args.reporting_col, args.seasoning_col):
        if col not in panel.columns:
            raise SystemExit(
                f"Derive mode needs '{col}'. Available: {list(panel.columns)} "
                f"(or pass --origination-col).")
    rep = pd.to_datetime(panel[args.reporting_col]).dt.to_period("M")
    orig_period = rep - panel[args.seasoning_col].astype(int)
    per_loan = (
        pd.DataFrame({args.id_col: panel[args.id_col].to_numpy(), "op": orig_period})
        .groupby(args.id_col)["op"].min()           # constant per loan; min is defensive
    )
    return per_loan.dt.to_timestamp()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/Overall_2024_2025_all_months.parquet")
    ap.add_argument("--id-col", default="loan_id")
    ap.add_argument("--origination-col", default=None,
                    help="explicit origination-date column; omit to derive from reporting-seasoning")
    ap.add_argument("--reporting-col", default="reporting_date")
    ap.add_argument("--seasoning-col", default="seasoning_months")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--fractions", default="0.8,0.1,0.1", help="train,val,test (sum to 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    fractions = tuple(float(x) for x in args.fractions.split(","))
    in_path = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    panel = pd.read_parquet(in_path)
    if args.id_col not in panel.columns:
        raise SystemExit(f"Column '{args.id_col}' not in panel. Available: {list(panel.columns)}")

    origination = _loan_origination(panel, args)
    mode = args.origination_col or f"derived({args.reporting_col}-{args.seasoning_col})"
    print(f"Origination key: {mode}  "
          f"({str(origination.min().date())} -> {str(origination.max().date())})")

    assignment = temporal_loan_split(origination, fractions=fractions)
    split_series = pd.Series(assignment, name="split")

    # write per-split parquets — a loan's entire history travels together
    panel = panel.assign(_split=panel[args.id_col].map(assignment))
    counts: dict[str, int] = {}
    ranges: dict[str, list[str]] = {}
    for s in SPLITS:
        sub = panel[panel["_split"] == s].drop(columns="_split")
        sub.to_parquet(out / f"{s}.parquet", index=False)
        orig_in = origination[split_series[split_series == s].index]
        counts[s] = int(split_series.eq(s).sum())
        ranges[s] = [str(orig_in.min().date()), str(orig_in.max().date())]
        print(f"  {s:>5}: {counts[s]:>7,} loans  {len(sub):>10,} rows  "
              f"origination {ranges[s][0]} -> {ranges[s][1]}")

    # loan_id -> split
    split_series.rename_axis(args.id_col).reset_index().to_csv(out / "splits.csv", index=False)

    # audit manifest
    meta = {
        "seed": args.seed,
        "split_date": date.today().isoformat(),
        "source_panel": str(in_path),
        "source_panel_sha256": _sha256(in_path),
        "n_loans": counts,
        "split_criterion": "loan_stratified_temporal_origination",
        "origination_key": mode,
        "fractions": list(fractions),
        "id_col": args.id_col,
        "origination_range": ranges,
        "code_commit": _git_commit(),
    }
    (out / "splits.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}/splits.csv and {out}/splits.meta.json")


if __name__ == "__main__":
    main()
