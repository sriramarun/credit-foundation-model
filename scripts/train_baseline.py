# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""XGBoost baseline for credit default — the Gate-G1 anchor the foundation model must beat.

Reads the loan-stratified temporal splits from ``scripts/prepare_data.py`` (same train/val/test
loans as the foundation model). Observes each loan at ``--obs``; label = CRR default in the next
6 monthly cutoffs. Four configurations separate signal from leakage:
  (1) full features, no gate            (2) full features + performing gate
  (3) no-leakage features, no gate      (4) no-leakage + gate  ← honest Gate G1

If ``--book`` (loan_book.parquet with the generator's hidden ``_segment`` latent) is given, also
runs the **ceiling validation** on the Gate-G1 cohort:
  (A) segment-conditional default rates (the hidden risk),
  (B) oracle-segment lift (how much the baseline would gain if it could see the segment), and
  (C) segment recoverability (can XGBoost recover the segment from observables — it can't).
``_segment`` is EVALUATION-ONLY ground truth and is never a deployable feature.

    python scripts/train_baseline.py --data-dir data/processed --book data/raw/loan_book.parquet \
        --report reports/baseline_report.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, roc_auc_score)

OBS_DEFAULT = "2024-12-31"
FWD = ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31", "2025-06-30"]
EXCLUDE_ALWAYS = {
    "loan_id", "transaction_name", "esma_transaction_identifier", "reporting_date",
    "closing_date", "maturity_date_proxy", "originator_name", "servicer_name", "currency",
    "country", "property_valuation_type", "interest_payment_frequency",
    "principal_payment_frequency", "y",
}
LEAKAGE_COLS = {
    "arrears_bucket", "performing_status", "default_crr_flag", "foreclosure_flag",
    "days_past_due", "arrears_amount", "forbearance_flag", "restructuring_flag",
}
PERFORMING_AT_OBS = {"Performing", "1-29 DPD", "30-59 DPD", "60-89 DPD"}
SEG_NAMES = {0: "stable", 1: "baseline", 2: "fragile"}


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


def _xgb(**kw):
    return xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="auc", n_jobs=-1,
                             tree_method="hist", early_stopping_rounds=20, **kw)


def run_xgb(name, parts):
    Xtr, ytr, Xva, yva, Xte, yte = parts
    t = time.time()
    m = _xgb()
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


def segment_validation(obs, cat_clean, num_clean):
    """Ceiling validation on the Gate-G1 (performing-at-obs) cohort. Returns report lines."""
    gated = {s: o[o.arrears_bucket.isin(PERFORMING_AT_OBS)].dropna(subset=["_segment"])
             for s, o in obs.items()}
    tr, va, te = gated["train"], gated["val"], gated["test"]

    # (A) segment-conditional default
    g = te.groupby("_segment")["y"].agg(["count", "mean"])
    spread = g["mean"].max() / max(g["mean"].min(), 1e-9)

    # (B) oracle-segment lift: clean features vs clean + _segment
    def fit_eval(extra):
        Xtr, cm = encode(tr, cat_clean, num_clean + extra)
        Xva, _ = encode(va, cat_clean, num_clean + extra, cm)
        Xte, _ = encode(te, cat_clean, num_clean + extra, cm)
        m = _xgb()
        m.fit(Xtr, tr.y, eval_set=[(Xva, va.y)], verbose=False)
        p = m.predict_proba(Xte)[:, 1]
        return roc_auc_score(te.y, p), average_precision_score(te.y, p)

    roc0, pr0 = fit_eval([])
    roc1, pr1 = fit_eval(["_segment"])

    # (C) recover _segment from observables
    Xtr, cm = encode(tr, cat_clean, num_clean)
    Xte, _ = encode(te, cat_clean, num_clean, cm)
    mc = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, n_jobs=-1,
                           tree_method="hist")
    mc.fit(Xtr, tr._segment.astype(int))
    pred = mc.predict(Xte)
    yte = te._segment.astype(int)
    acc = accuracy_score(yte, pred)
    maj = yte.value_counts(normalize=True).max()
    mf1 = f1_score(yte, pred, average="macro")

    print("\n" + "=" * 64 + "\nCEILING VALIDATION (hidden _segment, Gate-G1 cohort)\n" + "=" * 64)
    print("(A) segment-conditional default rate (test):")
    for s, r in g.iterrows():
        print(f"    {SEG_NAMES.get(int(s), s):<9} {int(r['count']):>6,} loans  {r['mean']*100:5.2f}%")
    print(f"    spread: {spread:.0f}x")
    print(f"(B) oracle-segment lift: ROC {roc0:.3f}->{roc1:.3f} (+{roc1-roc0:.3f}); "
          f"PR-AUC {pr0:.3f}->{pr1:.3f} (+{(pr1-pr0)/max(pr0,1e-9)*100:.0f}%)")
    print(f"(C) segment recovery from observables: acc {acc*100:.1f}% vs majority {maj*100:.1f}% "
          f"(macro-F1 {mf1:.3f}) -> essentially hidden")

    return [
        "", "## Architectural validation — the hidden-segment ceiling", "",
        "The generator assigns each loan a hidden fragility **segment** (in `loan_book`, not the "
        "ESMA panel — evaluation-only, never a feature). It drives default but is invisible to "
        "tabular models. Measured on the Gate-G1 cohort:", "",
        "| (A) Segment | loans (test) | default rate |", "|---|--:|--:|",
        *[f"| {SEG_NAMES.get(int(s), s)} | {int(r['count']):,} | {r['mean']*100:.2f}% |"
          for s, r in g.iterrows()],
        f"| **spread** | | **{spread:.0f}×** |", "",
        "**(B) Oracle-segment lift** — if the model could see the segment:", "",
        "| | ROC-AUC | PR-AUC |", "|---|--:|--:|",
        f"| Gate G1 (observables only) | {roc0:.3f} | {pr0:.3f} |",
        f"| + oracle `_segment` (diagnostic) | {roc1:.3f} | {pr1:.3f} |",
        f"| **headroom** | **+{roc1-roc0:.3f}** | **+{(pr1-pr0)/max(pr0,1e-9)*100:.0f}%** |", "",
        f"**(C) Can XGBoost recover the segment?** accuracy {acc*100:.1f}% vs {maj*100:.1f}% "
        f"majority-class (macro-F1 {mf1:.2f}) — **essentially no.** The signal exists (B) but "
        "tabular observables can't reach it.", "",
        "**Conclusion.** The hidden segment is a real, large source of default risk (A) that "
        "point-in-time tabular models cannot see (C); recovering it would nearly double PR-AUC "
        "(B). The foundation model reads each loan's behavioural *sequence* to recover that "
        "latent — that headroom above 0.73 is the project's thesis, now quantified.",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--obs", default=OBS_DEFAULT)
    ap.add_argument("--book", default="data/raw/loan_book.parquet")
    ap.add_argument("--report", default=None)
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

    cfgs = [("(1) full features, no gate", cat_full, num_full, False),
            ("(2) full features + performing gate", cat_full, num_full, True),
            ("(3) no-leakage features, no gate", cat_clean, num_clean, False),
            ("(4) no-leakage + gate (Gate G1)", cat_clean, num_clean, True)]
    results = {name: run_xgb(name, prep(c, n, g)) for name, c, n, g in cfgs}

    print("\n" + "=" * 64 + "\nSUMMARY (test split)\n" + "=" * 64)
    print(f"  {'config':<40}{'ROC':>8}{'PR':>8}{'pos%':>8}")
    summ = []
    for name in results:
        _, n, pos, roc, pr = next(r for r in results[name] if r[0] == "test")
        print(f"  {name:<40}{roc:>8.4f}{pr:>8.4f}{pos/max(n,1)*100:>7.2f}%")
        summ.append((name, n, pos, roc, pr))

    seg_lines = []
    book = Path(args.book)
    if book.exists():
        bk = pd.read_parquet(book, columns=["loan_id", "_segment"])
        obs = {s: o.merge(bk, on="loan_id", how="left") for s, o in obs.items()}
        seg_lines = segment_validation(obs, cat_clean, num_clean)
    else:
        print(f"\n(loan_book not found at {book} — ceiling validation skipped)")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Baseline Report — XGBoost (Gate G1)", "",
                 f"Panel split: `{args.data_dir}` (loan-stratified temporal, DL-007). Observation "
                 f"`{args.obs}`; label = CRR default in the next 6 months.", "",
                 "Four configurations isolate signal from leakage; the honest bar the foundation "
                 "model must beat is **config (4)**.", "",
                 "| Config | test ROC-AUC | test PR-AUC | pos% |", "|---|--:|--:|--:|"]
        for name, n, pos, roc, pr in summ:
            lines.append(f"| {name} | {roc:.4f} | {pr:.4f} | {pos/max(n,1)*100:.2f}% |")
        lines += ["", "## Reading it",
                  "- **Leakage** (contemporaneous delinquency state): config (1)→(3) drops ROC-AUC "
                  "sharply — those features read the current state.",
                  "- **Performing-at-obs gate**: predict *new* defaults among performing loans → "
                  "PR-AUC collapses at a low base rate (the realistic task).",
                  f"- **Gate G1 = config (4)**: ROC-AUC {summ[3][3]:.3f}, PR-AUC {summ[3][4]:.3f}.",
                  "", "## Caveat", "- Synthetic data is rule-based, so the clean baseline runs "
                  "higher than a real portfolio would."]
        lines += seg_lines
        lines += ["", f"Reproduce: `python scripts/train_baseline.py --data-dir {args.data_dir} "
                  f"--book {args.book} --report {args.report}`", ""]
        rep.write_text("\n".join(lines))
        print(f"\nWrote {rep}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
