# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Generate ``notebooks/06_fannie_data_bible.ipynb`` — the Fannie Mae dataset 'bible'.

Kept as a builder (not a hand-written .ipynb) so the notebook is regenerated deterministically and
reviewed as plain Python. Run from anywhere::

    python notebooks/build_fannie_bible.py

The notebook itself reads ``reports/fannie_dataset_profile.json`` (produced by
``scripts/profile_fannie_dataset.py``); glossary + include/exclude lists are derived live from
``src/credit_fm/data/fannie_glossary.py`` and ``configs/fannie_mae/``.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "06_fannie_data_bible.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
    md(r"""
# The Fannie Mae Loan-Performance Data Bible

**One notebook that documents the entire dataset**: every column's meaning, which columns the model
is allowed to see (and which are held out as leakage), and the headline risk signal — the
**delinquency / default rate by year** across the whole loan book.

It reads a pre-computed statistics artifact so it renders instantly; regenerate that artifact with:

```bash
# representative 4% panel (fast; the deterministic hash sample is unbiased for rates)
python scripts/profile_fannie_dataset.py \
    --panel gs://sriram-credit-fm-data/output/raw/fannie_mae/panel_2000_2024.parquet \
    --out reports/fannie_dataset_profile.json

# OR the TRUE whole loan book, straight from the raw source
python scripts/profile_fannie_dataset.py \
    --raw-root gs://sriram-credit-fm-data/fannie_by_reporting \
    --out reports/fannie_dataset_profile.json --no-vintage
```

**Contents**
1. Setup &amp; overview
2. Column glossary (all 113 source fields + 6 derived)
3. Columns *included* in training vs *excluded* / *leakage*
4. Delinquency &amp; default rate by year
5. Per-column statistics
6. Notes &amp; caveats
"""),

    # ---------------------------------------------------------------- setup
    md("## 1. Setup &amp; overview"),
    code(r"""
import importlib.util, json, re
from pathlib import Path
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 100)

# find the repo root (walk up until we see configs/)
ROOT = Path.cwd()
while not (ROOT / "configs" / "fannie_mae").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
assert (ROOT / "configs" / "fannie_mae").exists(), "run inside the credit-foundation-model repo"

# import the glossary module directly (avoids importing credit_fm/__init__, which pulls in torch)
_gspec = importlib.util.spec_from_file_location(
    "fannie_glossary", ROOT / "src" / "credit_fm" / "data" / "fannie_glossary.py")
G = importlib.util.module_from_spec(_gspec); _gspec.loader.exec_module(G)

# load the pre-computed statistics profile (may be absent on a fresh checkout)
PROFILE_PATH = ROOT / "reports" / "fannie_dataset_profile.json"
PROFILE = json.loads(PROFILE_PATH.read_text()) if PROFILE_PATH.exists() else None
print("glossary fields:", len(G.ALL_FIELDS), "| profile:",
      "loaded" if PROFILE else f"MISSING -> {PROFILE_PATH} (run profile_fannie_dataset.py)")
"""),
    code(r"""
if PROFILE:
    print(f"source     : {PROFILE['source']}  ({PROFILE['source_kind']})")
    print(f"generated  : {PROFILE['generated_utc']}")
    print(f"rows       : {PROFILE['n_rows']:,}  (loan-months)")
    print(f"loans      : {PROFILE['n_loans']}")
    print(f"columns    : {PROFILE['n_columns']}")
    print(f"reporting  : {PROFILE['reporting_range'][0]} .. {PROFILE['reporting_range'][1]}")
    print(f"origination: {PROFILE['origination_range'][0]} .. {PROFILE['origination_range'][1]}")
else:
    print("No profile artifact yet — sections 4 and 5 will be empty until you generate it.")
"""),

    # ---------------------------------------------------------------- glossary
    md(r"""
## 2. Column glossary

Every field in the dataset, condensed from the official *Single-Family Loan Performance Dataset and
Credit Risk Transfer — Glossary and File Layout* (© 2026 Fannie Mae). `position` is the 1-based
field position in the published layout; the last 6 rows (`position = derived`) are columns our
ingest step adds so the rest of the pipeline stays asset-generic.
"""),
    code(r"""
def glossary_frame(fields):
    rows = []
    for name, (pos, plain, dtype, desc, enums) in fields.items():
        rows.append({"position": pos if pos is not None else "derived", "column": name,
                     "name": plain, "type": dtype, "description": desc,
                     "enumerations": enums or ""})
    df = pd.DataFrame(rows)
    # raw fields first (by position), derived last
    df["_sort"] = [p if isinstance(p, int) else 10_000 + i for i, p in enumerate(df["position"])]
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

GLOSSARY = glossary_frame(G.ALL_FIELDS)
print(f"{len(GLOSSARY)} fields ({len(G.RAW_FIELDS)} source + {len(G.DERIVED_FIELDS)} derived)")
GLOSSARY
"""),
    md("**The 6 derived columns** (labels &amp; keys the pipeline computes) in detail:"),
    code(r"""
glossary_frame(G.DERIVED_FIELDS)
"""),

    # ---------------------------------------------------------------- include / exclude
    md(r"""
## 3. Columns *included* in training vs *excluded* / *leakage*

The included/excluded split is **derived live** from `configs/fannie_mae/raw_schema.yaml` (all 113
source fields) and `configs/fannie_mae/baseline.yaml` (the `exclude` / `leakage` lists + the
id/time/label/gate roles), so this table can never drift from the config the model actually uses.

* **Model features** — everything not excluded and not leakage.
* **Excluded (non-features)** — ids, raw dates (superseded by derived ISO dates), high-cardinality
  geo, and non-tabular strings.
* **Leakage** — outcome / contemporaneous-state / post-default servicing columns. Using any of
  these would let the model peek at the answer, so they are dropped and the task is gated to
  loans *performing at the observation date* (predict **new** defaults).
"""),
    code(r"""
def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())

schema = load_yaml(ROOT / "configs" / "fannie_mae" / "raw_schema.yaml")
base = load_yaml(ROOT / "configs" / "fannie_mae" / "baseline.yaml")

all_cols = [c["name"] for c in schema["columns"]]
exclude = set(base.get("exclude", []))
leakage = set(base.get("leakage", []))
roles = {base.get("id_col"), base.get("time_col"), base.get("label_col"), base.get("gate_col")}
roles.discard(None)

features = [c for c in all_cols if c not in exclude and c not in leakage and c not in roles]
excluded = [c for c in all_cols if c in exclude]
leak_raw = [c for c in all_cols if c in leakage]

print(f"source fields          : {len(all_cols)}")
print(f"  model features       : {len(features)}")
print(f"  excluded (non-feature): {len(excluded)}")
print(f"  leakage (held out)   : {len(leak_raw)}")
print(f"  role cols (id/time/label/gate): {sorted(roles)}")
"""),
    md("### 3a. Model features (what the model *is* trained on)"),
    code(r"""
pd.DataFrame([{"column": c, "name": G.ALL_FIELDS[c][1], "type": G.ALL_FIELDS[c][2],
               "description": G.ALL_FIELDS[c][3]} for c in features]).reset_index(drop=True)
"""),
    md("### 3b. Excluded non-feature columns (ids, raw dates, geo, non-tabular)"),
    code(r"""
pd.DataFrame([{"column": c, "name": G.ALL_FIELDS[c][1], "why": G.ALL_FIELDS[c][3]}
              for c in excluded]).reset_index(drop=True)
"""),
    md("### 3c. Leakage columns (outcome / contemporaneous / post-default — held out)"),
    code(r"""
pd.DataFrame([{"column": c, "name": G.ALL_FIELDS[c][1], "why": G.ALL_FIELDS[c][3]}
              for c in leak_raw]).reset_index(drop=True)
"""),

    # ---------------------------------------------------------------- delinquency by year
    md(r"""
## 4. Delinquency &amp; default rate by year

Two complementary views of credit risk across the whole book:

* **By reporting (calendar) year** — of all loan-months observed in year *Y*, the share that were
  30+ days past due, 180+ days past due (**D180**, our default threshold), in a **default_event**
  (D180 *or* a credit-event zero-balance code), and **performing**.
* **By vintage (origination year)** — of all loans *originated* in year *Y*, the share that **ever**
  hit a default_event over their observed life (loan-level lifetime default rate).
"""),
    code(r"""
if PROFILE and PROFILE["delinquency_by_reporting_year"]:
    dlq = pd.DataFrame(PROFILE["delinquency_by_reporting_year"]).set_index("year")
    display(dlq)
else:
    dlq = None
    print("No profile — generate reports/fannie_dataset_profile.json first.")
"""),
    code(r"""
if dlq is not None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dlq.index, dlq["dpd30_plus_pct"], marker="o", label="30+ days past due")
    ax.plot(dlq.index, dlq["d180_plus_pct"], marker="s", label="180+ days (D180)")
    ax.plot(dlq.index, dlq["default_event_pct"], marker="^", label="default_event (D180 or credit ZBC)")
    ax.set_xlabel("reporting year"); ax.set_ylabel("% of loan-months")
    ax.set_title("Delinquency & default rate by reporting year — whole loan book")
    ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout(); plt.show()
"""),
    code(r"""
if PROFILE and PROFILE["vintage_default_by_origination_year"]:
    vint = pd.DataFrame(PROFILE["vintage_default_by_origination_year"]).set_index("origination_year")
    display(vint)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(vint.index, vint["lifetime_default_pct"], color="#b3122f")
    ax.set_xlabel("origination (vintage) year"); ax.set_ylabel("% of loans ever in default")
    ax.set_title("Lifetime default rate by vintage — whole loan book")
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout(); plt.show()
else:
    print("No vintage table (profile missing or built with --no-vintage).")
"""),

    md(r"""
### 4d. Is the 4% sample representative? — 4% panel vs 100% book

The pretraining panel is a **deterministic 4% hash sample on `loan_id`** (whole loan histories kept
or dropped together). This section proves that sampling doesn't distort the risk signal: it overlays
the sample's delinquency curve on the whole book's. Provide a second profile of the **full** book to
activate it:

```bash
python scripts/profile_fannie_dataset.py \
    --raw-root gs://sriram-credit-fm-data/fannie_by_reporting \
    --out reports/fannie_dataset_profile_full.json --delinquency-only --no-vintage --no-loan-count
```

The **pooled** (loan-month-weighted) default rate is the robust headline; per-year gaps in thin years
are just sampling noise.
"""),
    code(r"""
FULL_PATH = ROOT / "reports" / "fannie_dataset_profile_full.json"
if PROFILE and FULL_PATH.exists():
    CMP = importlib.util.spec_from_file_location("cmp", ROOT / "scripts" / "compare_profiles.py")
    cmp = importlib.util.module_from_spec(CMP); CMP.loader.exec_module(cmp)
    full = json.loads(FULL_PATH.read_text())
    LA, LB = "4% sample", "100% book"
    yt = cmp._year_table(PROFILE, full, LA, LB)
    pa, pb = cmp._pooled(PROFILE), cmp._pooled(full)
    print(f"pooled default rate — {LA}: {pa['default_event_pct']}%   "
          f"{LB}: {pb['default_event_pct']}%   "
          f"(Δ {round(pa['default_event_pct'] - pb['default_event_pct'], 4)} pp, "
          f"{cmp._rel(pa['default_event_pct'], pb['default_event_pct'])}% rel)")
    display(yt[[f"default_event_pct__{LA}", f"default_event_pct__{LB}",
                "default_event_pct__diff_pp", "default_event_pct__diff_rel%"]])
else:
    yt = None
    print("Provide reports/fannie_dataset_profile_full.json (see the command above) to compare.")
"""),
    code(r"""
if yt is not None:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    yr = yt.index
    ax1.plot(yr, yt["default_event_pct__4% sample"], marker="o", label="4% sample")
    ax1.plot(yr, yt["default_event_pct__100% book"], marker="s", label="100% book")
    ax1.set_title("Default rate by year — sample vs book"); ax1.set_xlabel("reporting year")
    ax1.set_ylabel("default_event %"); ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.bar(yr, yt["default_event_pct__diff_pp"], color="#6a51a3")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("Gap: sample − book (percentage points)"); ax2.set_xlabel("reporting year")
    ax2.set_ylabel("Δ pp"); ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()
"""),

    # ---------------------------------------------------------------- per-column stats
    md(r"""
## 5. Per-column statistics

Computed by `scripts/profile_fannie_dataset.py` in a single memory-bounded streaming pass over the
dataset. Numeric columns report min/mean/std and quantiles (quantiles from a 200k reservoir sample);
categorical columns report their top values; distinct counts are exact up to a 200k cap.
"""),
    code(r"""
def stats_frames(profile):
    numeric, categ = [], []
    for name, s in profile["columns"].items():
        base = {"column": name, "n": s["n"], "nulls": s["nulls"], "null_%": s["null_pct"],
                "n_unique": s["n_unique"]}
        if s["kind"] == "numeric" and s.get("numeric"):
            numeric.append({**base, **s["numeric"]})
        else:
            top = s.get("top_values") or []
            base["min"], base["max"] = s.get("min"), s.get("max")
            base["top_values"] = ", ".join(f"{v}={p}%" for v, c, p in top[:6])
            categ.append(base)
    return (pd.DataFrame(numeric).set_index("column") if numeric else pd.DataFrame(),
            pd.DataFrame(categ).set_index("column") if categ else pd.DataFrame())

if PROFILE:
    NUM, CAT = stats_frames(PROFILE)
else:
    NUM = CAT = pd.DataFrame()
    print("No profile — nothing to show.")
"""),
    md("### 5a. Numeric columns"),
    code("NUM if not NUM.empty else 'no numeric columns / no profile'"),
    md("### 5b. Categorical &amp; date columns (top values)"),
    code("CAT if not CAT.empty else 'no categorical columns / no profile'"),
    md("### 5c. Missingness — most-null columns"),
    code(r"""
if PROFILE and PROFILE["columns"]:
    miss = pd.DataFrame([{"column": n, "null_%": s["null_pct"]}
                         for n, s in PROFILE["columns"].items()])
    miss = miss.sort_values("null_%", ascending=False).set_index("column")
    display(miss.head(30))
    import matplotlib.pyplot as plt
    top = miss[miss["null_%"] > 0].head(25)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(11, max(3, 0.32 * len(top))))
        ax.barh(top.index[::-1], top["null_%"][::-1], color="#4c72b0")
        ax.set_xlabel("% null"); ax.set_title("Most-missing columns")
        plt.tight_layout(); plt.show()
"""),

    # ---------------------------------------------------------------- caveats
    md(r"""
## 6. Notes &amp; caveats

* **Sampling & representativeness.** The default profile runs on the ingested **4% panel**, a
  *deterministic hash sample on `loan_id`* — whole loan histories are kept or dropped together, so
  observed rates are **unbiased estimates** of the whole book. Point `--raw-root` at the raw source
  for exact whole-book numbers.
* **Label definition.** `default_event = (dlq_num >= 6, i.e. D180) OR zero_balance_code ∈
  {02, 03, 09, 15}`. `is_performing` gates the task to loans current at the observation date so the
  model predicts **new** defaults, not ones already in progress.
* **Leakage discipline.** Section 3c columns are never features. Splits are **by `loan_id`** (never
  by row) and temporal by **origination date** — see `docs/data/fannie_mae.md` and the decision log.
* **Unknown delinquency.** `current_loan_delinquency_status = 'XX'` (or blank after removal) becomes
  `dlq_num = <NA>`; every downstream consumer resolves it with `.fillna(False)`.
* **Field availability drifts over time.** Some fields are populated only after certain releases
  (e.g. `*_classic_fico` from Dec-2025, several loss/servicing fields post-2020) — high null % on
  those columns is expected, not a data error.
"""),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    OUT.write_text(nbf.writes(nb))
    print(f"wrote {OUT}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
