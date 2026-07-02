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

Config-driven (recipe: ``configs/fannie_mae/classify.yaml``). Run on the TRAIN split so
cardinality stats stay train-only::

    python scripts/classify_schema.py -c configs/fannie_mae/classify.yaml
    python scripts/classify_schema.py -c configs/fannie_mae/classify.yaml \
        --out configs/fannie_mae/tokenizer.yaml --drop '[days_past_due, construction_year_bucket]'
"""

from __future__ import annotations

import pandas as pd

from credit_fm.data.schema import classify_fields, find_redundant
from credit_fm.utils import storage
from credit_fm.utils.config import parse_cli, summarize

TYPE_GROUP = {"numeric": "numeric", "categorical": "categorical",
              "bucket": "categorical", "flag": "flags"}


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/fannie_mae/classify.yaml")
    print(f"config: {cfg.config_path}\n"
          f"{summarize(cfg, 'input', 'id_col', 'time_col', 'drop', 'out')}", flush=True)

    storage.ensure_auth(cfg.input, cfg.key)
    try:
        df = storage.read_parquet(cfg.input)
    except FileNotFoundError:
        raise SystemExit(f"{cfg.input} not found — run scripts/prepare_data.py first "
                         "(or override --input).") from None
    info = classify_fields(df, id_col=cfg.id_col, time_col=cfg.time_col)
    red = find_redundant(df, info, id_col=cfg.id_col)

    rep = pd.DataFrame(info).T[["role", "type", "n_unique", "null_frac"]]
    print(f"{cfg.input}: {len(df):,} rows, {df.shape[1]} cols\n")
    print(rep.to_string())
    print("\nrole:", rep.role.value_counts().to_dict())

    extra = [str(c).strip() for c in (cfg.get_path("drop") or []) if str(c).strip()]
    print("\nSAFE auto-drop:", red["safe"])
    print("REVIEW candidates (add to the recipe's drop: list to drop):")
    for c, why in sorted(red["review"].items()):
        mark = "DROP" if c in extra else "keep"
        print(f"  [{mark}] {c}  {why}")

    out = cfg.get_path("out")
    if not out:
        return

    drop_const = [c for c, d in info.items() if d["type"] == "constant"]
    reasons = {**red["safe"], **{c: red["review"].get(c, "user drop list") for c in extra}}
    drop_redundant = sorted(reasons, key=lambda c: list(df.columns).index(c))
    dropped = set(drop_const) | set(drop_redundant)

    def group(role):
        g: dict[str, list[str]] = {"numeric": [], "categorical": [], "flags": []}
        for c, d in info.items():
            if d["role"] == role and c not in dropped and d["type"] in TYPE_GROUP:
                g[TYPE_GROUP[d["type"]]].append(c)
        return {k: v for k, v in g.items() if v}

    cmd = (f"python scripts/classify_schema.py -c {cfg.config_path} --out {out}"
           + (f" --drop '[{', '.join(extra)}]'" if extra else ""))
    lines = [
        "# SPDX-License-Identifier: Apache-2.0",
        "# Generated reproducibly by scripts/classify_schema.py — DO NOT hand-edit.",
        f"# Regenerate: {cmd}",
        f"# Vocab + numeric bins fit on {cfg.input} ONLY (decision DL-008).",
        "schema: esma_annex2",
        f"id_col: {cfg.id_col}",
        f"time_col: {cfg.time_col}",
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
    storage.write_text("\n".join(lines), out)
    n_feat = sum(len(v) for role in ("static", "dynamic") for v in group(role).values())
    print(f"\nWrote {out}: {n_feat} features "
          f"({len(drop_const)} constant + {len(drop_redundant)} redundant dropped)")


if __name__ == "__main__":
    main()
