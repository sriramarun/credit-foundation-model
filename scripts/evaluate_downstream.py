# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Downstream verdict — do the FM's ``[USR]`` embeddings beat features-XGBoost at predicting default?

Given per-loan embeddings (from ``extract_embeddings.py``, observed at ``--cutoff``) and the full
processed panel, this:

1. builds the **forward-default label** — 1 if the loan defaults within ``--horizon-months`` *after*
   the cutoff (the same D180 / zero-balance event the baseline uses), 0 otherwise;
2. builds the **baseline feature snapshot** as-of the cutoff (the tokenizer's profile+event fields —
   i.e. the *same information* the FM saw, so the comparison isolates *representation*);
3. splits loans **disjointly** (hash on id), then trains a head three ways —
   **features / FM embeddings / combined** (XGBoost) plus a **linear probe** on the embeddings —
   and reports **ROC-AUC / PR-AUC** on the held-out loans.

The winner is decided on the held-out set: if *embeddings* (or *combined*) beats *features*, the FM
adds signal beyond the raw as-of-cutoff features. (For a calendar-OOT verdict vs the 0.757/0.784
bars, run this on a multi-year panel with a train-years/test-years split — see the runbook.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from credit_fm.utils import storage


def forward_default_loans(panel, id_col, time_col, label_col, cutoff, horizon_months):
    """Loan ids that record a default in ``(cutoff, cutoff + horizon]`` — the forward label = 1 set."""
    lo = pd.to_datetime(cutoff)
    hi = lo + pd.DateOffset(months=horizon_months)
    dt = pd.to_datetime(panel[time_col], errors="coerce")
    hit = panel[(dt > lo) & (dt <= hi) & panel[label_col].fillna(False).astype(bool)]
    return set(hit[id_col])


def features_asof(panel, id_col, time_col, cols, cutoff):
    """Each loan's most-recent row at/<= cutoff, restricted to the feature columns (the snapshot)."""
    hist = panel[pd.to_datetime(panel[time_col], errors="coerce") <= pd.to_datetime(cutoff)]
    snap = hist.sort_values(time_col).groupby(id_col).tail(1).set_index(id_col)
    return snap[[c for c in cols if c in snap.columns]]


def encode_features(df, cats, nums, cmap=None):
    """Map categoricals to ints (fit on train), coerce numerics — returns (X, cmap)."""
    df = df.copy()
    fit = cmap is None
    cmap = cmap or {}
    for c in cats:
        if fit:
            cmap[c] = {v: i for i, v in enumerate(pd.Series(df[c].dropna().unique()))}
        df[c] = df[c].map(cmap[c]).astype("float")
    for c in nums:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[cats + nums], cmap


def _xgb(device):
    return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
                             device=device, n_jobs=-1)


def _score(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, help="per-loan embeddings parquet (extract step)")
    ap.add_argument("--panel", required=True, help="full processed monthly panel (for label+features)")
    ap.add_argument("--config", default="configs/fannie_mae/tokenizer.yaml",
                    help="yaml with id_col/time_col + profile/event field lists")
    ap.add_argument("--cutoff", required=True, help="observation date used for the embeddings")
    ap.add_argument("--horizon-months", type=int, default=12)
    ap.add_argument("--label-col", default="default_event")
    ap.add_argument("--test-frac", type=float, default=0.3, help="held-out loan fraction (hash split)")
    ap.add_argument("--device", default="cuda", help="xgboost device: cuda|cpu")
    ap.add_argument("--report", default=None)
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    id_col, time_col = cfg["id_col"], cfg["time_col"]
    cats = list(cfg["profile"]["categorical"]) + list(cfg["event"]["categorical"])
    nums = list(cfg["profile"]["numeric"]) + list(cfg["event"]["numeric"])

    storage.ensure_auth(args.embeddings, args.key)
    emb = storage.read_parquet(args.embeddings)
    panel = storage.read_parquet(args.panel)
    ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    print(f"embeddings: {len(emb):,} loans x {len(ecols)} dims; panel {len(panel):,} rows", flush=True)

    # 1) forward-default label
    defaulted = forward_default_loans(panel, id_col, time_col, args.label_col, args.cutoff,
                                      args.horizon_months)
    emb["y"] = emb[id_col].isin(defaulted).astype(int)

    # 2) baseline feature snapshot as-of cutoff
    feats = features_asof(panel, id_col, time_col, cats + nums, args.cutoff)
    df = emb.merge(feats, left_on=id_col, right_index=True, how="left").reset_index(drop=True)

    # 3) loan-disjoint split (hash on id — reproducible)
    is_test = pd.util.hash_pandas_object(df[id_col], index=False) % 100 < int(args.test_frac * 100)
    tr, te = df[~is_test], df[is_test]
    print(f"split: train {len(tr):,} ({tr.y.mean()*100:.2f}% default) | "
          f"test {len(te):,} ({te.y.mean()*100:.2f}% default)", flush=True)

    Xtr_f, cmap = encode_features(tr, cats, nums)
    Xte_f, _ = encode_features(te, cats, nums, cmap)
    Xtr_e, Xte_e = tr[ecols].to_numpy(), te[ecols].to_numpy()
    Xtr_c = np.hstack([Xtr_f.to_numpy(), Xtr_e])
    Xte_c = np.hstack([Xte_f.to_numpy(), Xte_e])
    ytr, yte = tr.y.to_numpy(), te.y.to_numpy()

    results = []
    results.append(("features (XGB)", *_score(_xgb(args.device), Xtr_f, ytr, Xte_f, yte)))
    results.append(("FM embeddings (XGB)", *_score(_xgb(args.device), Xtr_e, ytr, Xte_e, yte)))
    results.append(("combined (XGB)", *_score(_xgb(args.device), Xtr_c, ytr, Xte_c, yte)))
    sc = StandardScaler().fit(Xtr_e)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    results.append(("FM embeddings (linear probe)",
                    *_score(lr, sc.transform(Xtr_e), ytr, sc.transform(Xte_e), yte)))

    base_roc = results[0][1]
    print("\n=== Downstream: default within "
          f"{args.horizon_months}mo of {args.cutoff} (held-out loans) ===")
    print(f"  {'model':<30}{'ROC':>8}{'PR-AUC':>9}{'dROC vs feats':>15}")
    for name, roc, pr in results:
        print(f"  {name:<30}{roc:>8.4f}{pr:>9.4f}{roc-base_roc:>+15.4f}")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Fannie Mae — FM Downstream Eval (loan-holdout probe)", "",
            f"Observed at **{args.cutoff}**, label = default within **{args.horizon_months} months** "
            f"after. {len(df):,} performing loans; loan-disjoint {int(args.test_frac*100)}% held out. "
            "Features = tokenizer profile+event fields as-of cutoff (same info the FM saw).", "",
            "| model | ROC-AUC | PR-AUC | dROC vs features |", "|---|--:|--:|--:|",
            *[f"| {n} | {roc:.4f} | {pr:.4f} | {roc-base_roc:+.4f} |" for n, roc, pr in results],
            "", "## Read",
            "- If **FM embeddings** or **combined** beats **features**, the FM adds signal beyond the "
            "raw as-of-cutoff features — the thesis, on real loans.",
            "- This is a loan-holdout probe on one window, not the calendar-OOT vs 0.757/0.784 "
            "(which needs a multi-year panel + train-years/test-years split).",
        ]
        rep.write_text("\n".join(lines))
        print(f"\nWrote {rep}")


if __name__ == "__main__":
    main()
