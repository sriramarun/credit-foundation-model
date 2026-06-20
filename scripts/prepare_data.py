# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Split a raw credit panel into loan-stratified temporal train/val/test parquets.

Writes ``data/processed/{train,val,test}.parquet`` (the whole 24-cutoff history of a loan
stays in one split), a ``splits.csv`` (``loan_id -> split``), and ``splits.meta.json`` — a
reproducibility/audit trail (seed, source SHA-256, loan counts, origination ranges, commit).

Example:
    python scripts/prepare_data.py \
        --input data/raw/Overall_2024_2025_all_months.parquet \
        --id-col loan_id --origination-col origination_date
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/Overall_2024_2025_all_months.parquet")
    ap.add_argument("--id-col", default="loan_id")
    ap.add_argument("--origination-col", default="origination_date")
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
    for col in (args.id_col, args.origination_col):
        if col not in panel.columns:
            raise SystemExit(f"Column '{col}' not in panel. Available: {list(panel.columns)}")

    # one origination date per loan, then assign loans to splits
    origination = pd.to_datetime(panel.groupby(args.id_col)[args.origination_col].min())
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
        "fractions": list(fractions),
        "id_col": args.id_col,
        "origination_col": args.origination_col,
        "origination_range": ranges,
        "code_commit": _git_commit(),
    }
    (out / "splits.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out}/splits.csv and {out}/splits.meta.json")


if __name__ == "__main__":
    main()
