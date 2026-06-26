# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Fit the KVT tokenizer on a credit panel, save it, and write a token QA report (Milestone M1).

Reads a per-loan **monthly** panel (loan_id × reporting_date rows), fits the per-field
bucketers/categoricals on TRAIN only, serializes the tokenizer to JSON, then encodes a sample of
loans to report vocabulary size, sequence-length distribution, and token health (roundtrip,
unseen-category and missing rates).

Example:
    python scripts/train_tokenizer.py \
        --config configs/fannie_mae/tokenizer.yaml \
        --train  data/processed/train.parquet \
        --out    configs/fannie_mae/tokenizer.json \
        --report reports/fannie_tokenizer_report.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from credit_fm.tokenizer import KVTTokenizer
from credit_fm.tokenizer.vocabulary import SPECIAL_TOKENS
from credit_fm.utils import storage


def _prune_to_panel(cfg: dict, columns) -> list[str]:
    """Drop configured fields not present in the panel; return the dropped names."""
    present = set(columns)
    dropped: list[str] = []
    for branch in ("profile", "event"):
        for kind in ("numeric", "categorical"):
            fields = cfg.get(branch, {}).get(kind, [])
            cfg[branch][kind] = [f for f in fields if f in present]
            dropped += [f for f in fields if f not in present]
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fannie_mae/tokenizer.yaml")
    ap.add_argument("--train", default="data/processed/train.parquet",
                    help="per-loan monthly panel; local path or gs:///s3:// URL")
    ap.add_argument("--out", default="configs/fannie_mae/tokenizer.json")
    ap.add_argument("--report", default="reports/fannie_tokenizer_report.md")
    ap.add_argument("--qa-loans", type=int, default=2000)
    ap.add_argument("--max-fit-rows", type=int, default=None, help="optional row cap for fitting")
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    storage.ensure_auth(args.train, args.key)
    print(f"Loading {args.train} ...")
    panel = storage.read_parquet(args.train)

    dropped = _prune_to_panel(cfg, panel.columns)
    if dropped:
        print(f"  skipping {len(dropped)} configured fields absent from panel: "
              f"{dropped[:8]}{' ...' if len(dropped) > 8 else ''}")
    for req in (cfg["id_col"], cfg["time_col"], cfg["time_field"]):
        if req not in panel.columns:
            raise SystemExit(f"required column '{req}' missing from panel")

    fit_df = panel
    if args.max_fit_rows and len(panel) > args.max_fit_rows:
        fit_df = panel.sample(args.max_fit_rows, random_state=42)
    t0 = time.time()
    tok = KVTTokenizer(cfg).fit(fit_df)
    print(f"fit: {tok.vocab_size:,} vocab tokens on {len(fit_df):,} rows in {time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer -> {args.out}")

    # ---- QA on a sample of loans ----
    idc = cfg["id_col"]
    loan_ids = panel[idc].drop_duplicates()
    qa_ids = set(loan_ids.sample(min(len(loan_ids), args.qa_loans), random_state=1))
    sample = panel[panel[idc].isin(qa_ids)]
    lengths, unk, na, total, lossless, n_loans = [], 0, 0, 0, 0, 0
    for _, loan in sample.groupby(idc):
        toks = tok.tokens(loan)
        ids = [tok.vocabulary.encode(t) for t in toks]
        lossless += int(tok.decode(ids) == toks)
        lengths.append(len(ids))
        n_loans += 1
        for tk in toks:
            total += 1
            unk += int(tk.endswith("=UNK"))
            na += int(tk.endswith("=NA"))
    lens = np.array(lengths)

    def pct(x: int) -> float:
        return 100.0 * x / max(total, 1)

    n_pf = len(cfg["profile"]["numeric"]) + len(cfg["profile"]["categorical"])
    n_ev = len(cfg["event"]["numeric"]) + len(cfg["event"]["categorical"])
    lines = [
        "# Fannie Mae — KVT Tokenizer Report (M1)", "",
        f"Fitted on `{args.train}` ({len(fit_df):,} rows). Config `{args.config}`; "
        f"saved to `{args.out}`.", "",
        "## Vocabulary", "",
        f"- **{tok.vocab_size:,} tokens** ({len(SPECIAL_TOKENS)} special + field value tokens).",
        f"- **Profile** {n_pf} fields ({len(cfg['profile']['numeric'])} numeric / "
        f"{len(cfg['profile']['categorical'])} categorical); **Event** {n_ev} fields "
        f"({len(cfg['event']['numeric'])} numeric / {len(cfg['event']['categorical'])} categorical); "
        f"time field `{cfg['time_field']}`.",
        "", f"## Sequence length (QA sample: {n_loans:,} loans)", "",
        "| stat | tokens / loan |", "|---|--:|",
        f"| min | {int(lens.min())} |",
        f"| median | {int(np.median(lens))} |",
        f"| p95 | {int(np.percentile(lens, 95))} |",
        f"| max | {int(lens.max())} |",
        "", "## Token health", "",
        "| metric | value |", "|---|--:|",
        f"| roundtrip lossless | {100.0*lossless/max(n_loans,1):.1f}% of loans |",
        f"| unseen-category tokens (`=UNK`) | {pct(unk):.2f}% |",
        f"| missing tokens (`=NA`) | {pct(na):.2f}% |",
        "", "## Notes",
        "- Bins/categories fit on TRAIN only; unseen values map to `=UNK`, missing to `=NA`.",
        "- Roundtrip is token-level lossless (fused `field=value` tokens); numeric exact values are "
        "bucketed by design, so the QA target is losslessness + low OOV, not value reconstruction.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines))
    print(f"wrote {args.report}  (roundtrip {100.0*lossless/max(n_loans,1):.1f}%, "
          f"median {int(np.median(lens))} tokens/loan)")


if __name__ == "__main__":
    main()
