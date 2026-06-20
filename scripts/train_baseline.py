# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""XGBoost baseline for credit default — the Gate-G1 anchor the foundation model must beat.

Reads the loan-stratified temporal splits written by ``scripts/prepare_data.py`` (so the
baseline uses the SAME train/val/test loans as the foundation model). Observes each loan at
``--obs`` and predicts a CRR default in the next 6 monthly cutoffs.

Runs four configurations to separate signal from leakage:
  (1) full features, no gate            — inflated; reads the answer off current delinquency
  (2) full features + performing gate   — predict NEW defaults, but still leaky features
  (3) no-leakage features, no gate
  (4) no-leakage features + gate        — the honest Gate-G1 number

If ``--book`` (loan_book.parquet with ``_segment``/``_latent_fragility``) exists, also prints
segment-conditional default rates — the hidden latent that caps a point-in-time tabular model.

    python scripts/train_baseline.py --data-dir data/processed --report reports/baseline_report.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

OBS_DEFAULT = "2024-12-31"
FWD = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
EXCLUDE_ALWAYS = {
    "loan_id", "transaction_name", "esma_transaction_identifier", "reporting_date",
    "closing_date", "maturity_date_proxy", "originator_name", "servicer_name", "currency",
    "country", "property_valuation_type", "interest_payment_frequency",
    "principal_payment_frequency", "y",
}
# contemporaneous-state features that encode/reveal the label
LEAKAGE_COLS = {
    "arrears_bucket", "performing_status", "default_crr_flag", "foreclosure_flag",
    "days_past_due", "arrears_amount", "forbearance_flag", "restructuring_flag",
}
PERFORMING_AT_OBS = {"Performing", "1-29 DPD", "30-59 DPD", "60-89 DPD"}


def load_obs(path: Path, obs_date: str) -> pd.DataFrame:
    obs = pd.read_parquet(path, filters=[("reporting_date", "=", obs_date)])
    fut = pd.read_parquet(path, columns=["loan_id", "default_crr_flag"],
                          filters=[("reporting_date", "in", FWD)])
    bad = set(fut.loc[fut.default_crr_flag == "Y", "loan_id"])
    obs["y"] = obs.loan_id.isin(bad).astype(int)
    return obs


def encode(df, cat_cols, num_cols, cats_map=None):
    df = df.copy()
    fit = cats_map is None
    cats_map = cats_map or {}
    for c in cat_cols:
        if fit:
            cats_map[c] = {v: i for i, v in enumerate(pd.Series(df[c].dropna().unique()))}
        df[c] = df[c].map(cats_map[c]).astype("float")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cat_cols + num_cols], cats_map


def run_xgb(name, parts):
    Xtr, ytr, Xva, yva, Xte, yte = parts
    t = time.time()
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="auc", n_jobs=-1,
                          tree_method="hist", early_stopping_rounds=20)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    rows = []
    for lab, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        if y.sum() < 5:
            rows.append((lab, len(y), int(y.sum()), float("nan"), float("nan")))
            continue
        p = m.predict_proba(X)[:, 1]
        rows.append((lab, len(y), int(y.sum()), roc_auc_score(y, p), average_precision_score(y, p)))
    print(f"\n=== {name}  ({time.time()-t:.0f}s, {Xtr.shape[1]} feats) ===")
    print(f"  {'split':<6}{'n':>9}{'pos':>7}{'pos%':>8}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for lab, n, pos, roc, pr in rows:
        rs = f"{roc:.4f}" if not np.isnan(roc) else " n/a"
        ps = f"{pr:.4f}" if not np.isnan(pr) else " n/a"
        print(f"  {lab:<6}{n:>9,}{pos:>7,}{pos/max(n,1)*100:>7.2f}%{rs:>10}{ps:>10}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/processed", help="dir with {train,val,test}.parquet")
    ap.add_argument("--obs", default=OBS_DEFAULT)
    ap.add_argument("--book", default="data/raw/loan_book.parquet")
    ap.add_argument("--report", default=None, help="optional markdown report path")
    args = ap.parse_args()
    d = Path(args.data_dir)

    t0 = time.time()
    obs = {s: load_obs(d / f"{s}.parquet", args.obs) for s in ("train", "val", "test")}
    for s, o in obs.items():
        print(f"  {s}: {len(o):,} loans @ {args.obs}, default rate {o.y.mean()*100:.2f}%")

    tr = obs["train"]
    cat_full = [c for c in tr.columns if c not in EXCLUDE_ALWAYS and tr[c].dtype == object]
    num_full = [c for c in tr.columns if c not in EXCLUDE_ALWAYS and tr[c].dtype != object]
    cat_clean = [c for c in cat_full if c not in LEAKAGE_COLS]
    num_clean = [c for c in num_full if c not in LEAKAGE_COLS]
    print(f"  features: full={len(cat_full)+len(num_full)}  clean={len(cat_clean)+len(num_clean)}")

    def prep(cat, num, gated):
        def slc(o):
            return o[o.arrears_bucket.isin(PERFORMING_AT_OBS)] if gated else o
        Xtr, cm = encode(slc(obs["train"]), cat, num)
        Xva, _ = encode(slc(obs["val"]), cat, num, cm)
        Xte, _ = encode(slc(obs["test"]), cat, num, cm)
        return (Xtr, slc(obs["train"]).y.to_numpy(), Xva, slc(obs["val"]).y.to_numpy(),
                Xte, slc(obs["test"]).y.to_numpy())

    cfgs = [
        ("(1) full features, no gate", cat_full, num_full, False),
        ("(2) full features + performing gate", cat_full, num_full, True),
        ("(3) no-leakage features, no gate", cat_clean, num_clean, False),
        ("(4) no-leakage + gate (Gate G1)", cat_clean, num_clean, True),
    ]
    results = {name: run_xgb(name, prep(c, n, g)) for name, c, n, g in cfgs}

    print("\n" + "=" * 64 + "\nSUMMARY (test split)\n" + "=" * 64)
    print(f"  {'config':<40}{'ROC':>8}{'PR':>8}{'pos%':>8}")
    summ = []
    for name in results:
        _, n, pos, roc, pr = next(r for r in results[name] if r[0] == "test")
        print(f"  {name:<40}{roc:>8.4f}{pr:>8.4f}{pos/max(n,1)*100:>7.2f}%")
        summ.append((name, n, pos, roc, pr))

    book = Path(args.book)
    if book.exists():
        bk = pd.read_parquet(book, columns=["loan_id", "_segment"])
        if "_segment" in bk.columns:
            full = pd.concat([o[["loan_id", "y"]] for o in obs.values()])
            mg = bk.merge(full, on="loan_id", how="inner")
            print("\nSegment-conditional default rate (observed loans):")
            for s in sorted(mg["_segment"].dropna().unique()):
                sub = mg[mg["_segment"] == s]
                print(f"  segment {int(s)}: {len(sub):,} loans, default {sub.y.mean()*100:.2f}%")
    else:
        print(f"\n(loan_book not found at {book} — segment validation skipped)")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Baseline Report — XGBoost (Gate G1)", "",
            f"Panel split: `{args.data_dir}` (loan-stratified temporal, DL-007). "
            f"Observation `{args.obs}`; label = CRR default in the next 6 months.", "",
            "Four configurations isolate signal from leakage. The honest baseline the "
            "foundation model must beat is **config (4)**.", "",
            "| Config | test ROC-AUC | test PR-AUC | pos% |",
            "|--------|----:|----:|----:|",
        ]
        for name, n, pos, roc, pr in summ:
            lines.append(f"| {name} | {roc:.4f} | {pr:.4f} | {pos/max(n,1)*100:.2f}% |")
        lines += [
            "", "## Reading it",
            "- **Leakage** (contemporaneous `arrears_bucket`/`performing_status`/"
            "`default_crr_flag`/…): config (1)→(3) drops ROC-AUC sharply — those features read "
            "the current delinquency state.",
            "- **Performing-at-obs gate**: predict *new* defaults among currently-performing "
            "loans → PR-AUC collapses at a low base rate (the realistic, hard task).",
            f"- **Gate G1 = config (4)**: ROC-AUC {summ[3][3]:.3f}, PR-AUC {summ[3][4]:.3f}.",
            "", "## Caveats",
            "- Synthetic data is rule-based, so even the clean baseline is higher than a real "
            "portfolio would give.",
            "- Architectural-validation (segment latent ceiling) requires "
            "`loan_book.parquet` with `_segment`/`_latent_fragility`.",
            "", f"Reproduce: `python scripts/train_baseline.py --data-dir {args.data_dir} "
            f"--report {args.report}`", "",
        ]
        rep.write_text("\n".join(lines))
        print(f"\nWrote {rep}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
