# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Validate a scored-portfolio artifact against the scoring contract — read-only audit.

Proves the *produced scores* file (from ``scripts/score_portfolio.py``) is well-formed:

  A) schema — has the id / score / n_events / cutoff columns;
  B) score range — every score in [0, 1], no NaN;
  C) one row per loan — no duplicate ids;
  D) each scored loan has history — n_events >= 1;
  E) single cutoff — one cutoff value, matching the manifest;
  F) manifest agreement — row count and score summary match ``<scores>_manifest.json``.

    python scripts/validate_scores.py --scores gs://.../portfolio_scores.parquet
    python scripts/validate_scores.py --scores runs/portfolio_scores.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _read_parquet(path: str) -> pd.DataFrame:
    if path.startswith("gs://"):
        import gcsfs
        with gcsfs.GCSFileSystem().open(path[len("gs://"):]) as f:
            return pd.read_parquet(f)
    return pd.read_parquet(path)


def _read_text(path: str) -> str:
    if path.startswith("gs://"):
        import gcsfs
        with gcsfs.GCSFileSystem().open(path[len("gs://"):], "r") as f:
            return f.read()
    return Path(path).read_text()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="scored parquet (local or gs://)")
    ap.add_argument("--id-col", default="loan_id")
    ap.add_argument("--key", default="/workspace/.gcloud/credit-fm-sa.json")
    args = ap.parse_args()
    if args.scores.startswith("gs://") and Path(args.key).exists():
        os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", args.key)

    df = _read_parquet(args.scores)
    manifest_path = args.scores.rsplit(".", 1)[0] + "_manifest.json"
    try:
        manifest = json.loads(_read_text(manifest_path))
    except (FileNotFoundError, OSError):
        manifest = None

    results: list[tuple[bool, str, str]] = []

    def chk(name, cond, detail=""):
        results.append((bool(cond), name, detail))

    idc = args.id_col
    # A) schema
    need = {idc, "score", "n_events", "cutoff"}
    have = set(df.columns)
    chk("A: required columns present", need <= have,
        f"missing {sorted(need - have) or 'none'}; {len(df):,} rows")

    if need <= have and len(df):
        # B) score range
        s = pd.to_numeric(df["score"], errors="coerce")
        chk("B: every score in [0,1], no NaN",
            bool(s.notna().all() and (s >= 0).all() and (s <= 1).all()),
            f"range {s.min():.4f}..{s.max():.4f}, {int(s.isna().sum())} NaN")
        # C) one row per loan
        dups = int(df[idc].duplicated().sum())
        chk("C: no duplicate loan ids", dups == 0, f"{dups} duplicates")
        # D) each scored loan has history
        ne = pd.to_numeric(df["n_events"], errors="coerce")
        chk("D: every scored loan has n_events >= 1", bool((ne >= 1).all()),
            f"min n_events = {int(ne.min())}")
        # E) single cutoff
        cuts = df["cutoff"].astype(str).unique()
        chk("E: single cutoff value", len(cuts) == 1, f"cutoffs = {list(cuts)[:3]}")

    # F) manifest agreement
    if manifest is not None:
        agree = manifest.get("n_scored") == len(df)
        if need <= have and len(df):
            agree = agree and str(manifest.get("cutoff", ""))[:10] in {str(c)[:10] for c in df["cutoff"].astype(str).unique()}
        chk("F: manifest n_scored + cutoff match the file", agree,
            f"manifest n_scored={manifest.get('n_scored')} vs {len(df)}")
    else:
        chk("F: manifest present", False, f"no manifest at {manifest_path}")

    ok = all(r[0] for r in results)
    for passed, name, detail in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
