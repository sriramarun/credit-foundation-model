# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Extract per-loan ``[USR]`` embeddings from a pretrained checkpoint.

Runs a processed monthly panel (tokenizer schema) through the frozen model and writes one row per
loan: ``loan_id`` + the ``dim`` embedding columns (``e0..e{dim-1}``) + carried columns + ``n_events``.

Leakage control lives here: pass ``--cutoff`` to truncate each loan's history to reporting dates
``<= cutoff`` (so the embedding only "sees" the past), and ``--gate-col is_performing`` to keep only
loans performing at that cutoff (the "predict *new* defaults" gate). The downstream label (default
within the forward window) is computed later, in ``evaluate_downstream.py``, from the full panel.

Example (observe at Dec 2016, performing only):
    python scripts/extract_embeddings.py \
        --panel gs://.../output/processed/fannie_mae/run_2016_2017/train.parquet \
        --checkpoint gs://.../runs/m3_full.pt --tokenizer configs/fannie_mae/tokenizer.json \
        --cutoff 2016-12-31 --gate-col is_performing --carry origination_date \
        --out gs://.../embeddings/m3_dec2016_train.parquet --bf16
"""

from __future__ import annotations

import argparse
import time

import fsspec
import numpy as np
import pandas as pd
import torch

from credit_fm.data.collators import MLMCollator
from credit_fm.data.encode import encode_panel
from credit_fm.models import CreditFoundationModel
from credit_fm.tokenizer import KVTTokenizer
from credit_fm.utils import storage


def load_checkpoint(path: str, key: str):
    """Rebuild the model from a checkpoint (state_dict + config) and set it to eval mode."""
    storage.ensure_auth(path, key)
    with fsspec.open(path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = CreditFoundationModel(
        c["vocab_size"], c["n_field_types"], dim=c["dim"], n_heads=c["n_heads"],
        profile_layers=c["profile_layers"], event_layers=c["event_layers"],
        history_layers=c["history_layers"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), c


def observe_panel(panel: pd.DataFrame, id_col: str, time_col: str,
                  cutoff: str | None, gate_col: str | None) -> pd.DataFrame:
    """Truncate history to ``<= cutoff`` and (optionally) keep only loans passing the gate at cutoff."""
    if cutoff is not None:
        dt = pd.to_datetime(panel[time_col], errors="coerce")
        panel = panel[dt <= pd.to_datetime(cutoff)]
    if gate_col is not None:
        last = panel.sort_values(time_col).groupby(id_col).tail(1)     # most recent kept row per loan
        keep = set(last.loc[last[gate_col].fillna(False).astype(bool), id_col])
        panel = panel[panel[id_col].isin(keep)]
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, help="processed monthly panel; local or gs://")
    ap.add_argument("--checkpoint", required=True, help="pretrained checkpoint (.pt); local or gs://")
    ap.add_argument("--tokenizer", default="configs/fannie_mae/tokenizer.json")
    ap.add_argument("--out", required=True, help="output embeddings parquet; local or gs://")
    ap.add_argument("--cutoff", default=None, help="keep reporting dates <= this ISO date (no leakage)")
    ap.add_argument("--gate-col", default=None,
                    help="keep loans truthy in this col at cutoff (e.g. is_performing)")
    ap.add_argument("--carry", default="", help="comma-sep panel cols to carry per loan (first value)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="cap loans (smoke test)")
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = KVTTokenizer.load(args.tokenizer)
    model, cfg = load_checkpoint(args.checkpoint, args.key)
    model.to(device)
    dim = cfg["dim"]
    print(f"model: dim={dim}, {model.num_parameters()/1e6:.1f}M params on {device}", flush=True)

    storage.ensure_auth(args.panel, args.key)
    panel = storage.read_parquet(args.panel)
    panel = observe_panel(panel, tok.id_col, tok.time_col, args.cutoff, args.gate_col)
    if args.limit:
        keep = panel[tok.id_col].drop_duplicates().head(args.limit)
        panel = panel[panel[tok.id_col].isin(keep)]
    print(f"panel: {panel[tok.id_col].nunique():,} loans "
          f"(cutoff={args.cutoff}, gate={args.gate_col})", flush=True)

    if panel.empty:
        raise SystemExit("no loans after cutoff/gate — run the diagnostic: check "
                         "reporting_date range/format and is_performing values")
    carry = [c for c in args.carry.split(",") if c]
    carried = panel.groupby(tok.id_col)[carry].first() if carry else None

    shard = encode_panel(tok, panel)                                  # per-loan token arrays
    collate = MLMCollator(vocab_size=tok.vocab_size, mask=False)      # no masking — inference
    use_amp = args.bf16 and device.startswith("cuda")

    t0, embs = time.time(), []
    for start in range(0, len(shard), args.batch_size):
        rows = shard.iloc[start:start + args.batch_size]
        samples = []
        for _, r in rows.iterrows():
            s = {k: torch.tensor(r[k], dtype=torch.long)
                 for k in ("input_ids", "event_index", "field_type", "branch")}
            s["n_events"] = int(r["n_events"])
            samples.append(s)
        batch = collate(samples)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            embs.append(model.extract_embeddings(batch).float().cpu().numpy())
    emb = np.concatenate(embs, axis=0)

    out = pd.DataFrame({tok.id_col: shard[tok.id_col].to_numpy()})
    out["n_events"] = shard["n_events"].to_numpy()
    for i in range(dim):
        out[f"e{i}"] = emb[:, i]
    if carried is not None:
        out = out.merge(carried, left_on=tok.id_col, right_index=True, how="left")

    storage.write_parquet(out, args.out)
    print(f"wrote {len(out):,} loan embeddings (dim={dim}) -> {args.out}  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
