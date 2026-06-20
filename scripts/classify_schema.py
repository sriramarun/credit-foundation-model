# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Classify the panel's columns and generate a tokenizer config — reproducibly.

Pipeline (all data-driven, so re-running regenerates the same file):
  1. classify each column: role (id/static/dynamic) + value type.
  2. drop ``constant`` columns (no signal).
  3. drop ``safe`` redundancies auto-detected from the data (exact dups + numeric ``*_bucket``).
  4. drop any ``review`` functional-dependency candidates the user opts into via ``--drop``
     (printed as suggestions; kept by default so explicit signals aren't lost silently).
  5. emit profile (static) / event (dynamic) field lists, split by type.

Run on the TRAIN split so cardinality stats stay train-only:
    python scripts/classify_schema.py --input data/processed/train.parquet \
        --out configs/dutch_mortgages/tokenizer.yaml \
        --drop interest_only_flag,self_employed_flag,property_usage,buy_to_let_flag,days_past_due,primary_energy_demand_kwh_m2,construction_year_bucket
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credit_fm.data.schema import classify_fields, find_redundant

TYPE_GROUP = {"numeric": "numeric", "categorical": "categorical",
              "bucket": "categorical", "flag": "flags"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/processed/train.parquet")
    ap.add_argument("--id-col", default="loan_id")
    ap.add_argument("--time-col", default="reporting_date")
    ap.add_argument("--drop", default="", help="extra review-redundant columns to drop (comma list)")
    ap.add_argument("--out", default=None, help="write the tokenizer YAML to this path")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"{src} not found — run scripts/prepare_data.py first (or pass --input).")
    df = pd.read_parquet(src)
    info = classify_fields(df, id_col=args.id_col, time_col=args.time_col)
    red = find_redundant(df, info, id_col=args.id_col)

    rep = pd.DataFrame(info).T[["role", "type", "n_unique", "null_frac"]]
    print(f"{src}: {len(df):,} rows, {df.shape[1]} cols\n")
    print(rep.to_string())
    print("\nrole:", rep.role.value_counts().to_dict())

    extra = [c.strip() for c in args.drop.split(",") if c.strip()]
    print("\nSAFE auto-drop:", red["safe"])
    print("REVIEW candidates (pass via --drop to drop):")
    for c, why in sorted(red["review"].items()):
        mark = "DROP" if c in extra else "keep"
        print(f"  [{mark}] {c}  {why}")

    if not args.out:
        return

    drop_const = [c for c, d in info.items() if d["type"] == "constant"]
    reasons = {**red["safe"], **{c: red["review"].get(c, "user --drop") for c in extra}}
    drop_redundant = sorted(reasons, key=lambda c: list(df.columns).index(c))
    dropped = set(drop_const) | set(drop_redundant)

    def group(role):
        g: dict[str, list[str]] = {"numeric": [], "categorical": [], "flags": []}
        for c, d in info.items():
            if d["role"] == role and c not in dropped and d["type"] in TYPE_GROUP:
                g[TYPE_GROUP[d["type"]]].append(c)
        return {k: v for k, v in g.items() if v}

    cmd = (f"python scripts/classify_schema.py --input {args.input} "
           f"--out {args.out}" + (f" --drop {args.drop}" if args.drop else ""))
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated reproducibly by scripts/classify_schema.py — DO NOT hand-edit.",
        f"# Regenerate: {cmd}",
        "# Vocab + numeric bins fit on data/processed/train.parquet ONLY (decision DL-008).",
        "schema: esma_annex2",
        f"id_col: {args.id_col}",
        f"time_col: {args.time_col}",
        "num_buckets: 16",
        "zero_bucket: true",
        "",
        "drop_constant:        # single value across the panel — no signal",
    ]
    lines += [f"  - {c}" for c in drop_const]
    lines.append("\ndrop_redundant:       # exact-dup / numeric bucket (auto) + chosen functional deps")
    lines += [f"  - {c}    # {reasons[c]}" for c in drop_redundant]
    for role, key in (("static", "profile"), ("dynamic", "event")):
        lines.append(f"\n{key}:        # {role} fields")
        for sub, cols in group(role).items():
            lines.append(f"  {sub}:")
            lines += [f"    - {c}" for c in cols]
    lines += ["", "temporal:", "  reference: origination", "  log_seconds: true",
              "  cyclical: [month, quarter]", ""]
    Path(args.out).write_text("\n".join(lines))
    n_feat = sum(len(v) for role in ("static", "dynamic") for v in group(role).values())
    print(f"\nWrote {args.out}: {n_feat} features "
          f"({len(drop_const)} constant + {len(drop_redundant)} redundant dropped)")


if __name__ == "__main__":
    main()
