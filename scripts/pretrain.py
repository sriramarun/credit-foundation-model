# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Pretrain the credit foundation model (MLM) over encode-once shards.

Loads a frozen ``tokenizer.json`` (for ``vocab_size`` + ``n_field_types``), builds a
``CreditDataModule`` over the encoded shard directories, constructs ``CreditFoundationModel``, and
runs ``train_mlm``. Doubles as the **M2 toy run** (small ``--limit``/``--steps``/``--dim`` on one
GPU or CPU) and the seed of the **M3** full pretrain (scale up the same flags on 8x H100).

Example (toy, one GPU):
    python scripts/pretrain.py \
        --tokenizer configs/fannie_mae/tokenizer.json \
        --train-dir gs://sriram-credit-fm-data/output/encoded/fannie_mae/run_2016_2017/train \
        --val-dir   gs://sriram-credit-fm-data/output/encoded/fannie_mae/run_2016_2017/val \
        --limit 1000 --steps 100 --batch-size 32 --dim 256 --bf16 --out runs/toy.pt
"""

from __future__ import annotations

import argparse

import fsspec
import torch

from credit_fm.data import CreditDataModule
from credit_fm.models import CreditFoundationModel
from credit_fm.tokenizer import KVTTokenizer
from credit_fm.training import train_mlm
from credit_fm.utils import storage
from credit_fm.utils.reproducibility import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="configs/fannie_mae/tokenizer.json")
    ap.add_argument("--train-dir", required=True, help="encoded shard dir (local or gs://)")
    ap.add_argument("--val-dir", default=None)
    ap.add_argument("--out", default=None, help="checkpoint path (local or gs://)")
    # data
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap loans/split (toy run)")
    ap.add_argument("--num-workers", type=int, default=4)
    # model
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--profile-layers", type=int, default=3)
    ap.add_argument("--event-layers", type=int, default=5)
    ap.add_argument("--history-layers", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1, help="regularisation; 0 to disable")
    # optim
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # runtime
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    set_seed(args.seed)
    tok = KVTTokenizer.load(args.tokenizer)
    vocab_size, n_field_types = tok.vocab_size, len(tok.field_types)
    print(f"tokenizer: {vocab_size} tokens, {n_field_types} field types", flush=True)

    dm = CreditDataModule(args.train_dir, val_dir=args.val_dir, vocab_size=vocab_size,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          limit=args.limit, key=args.key)
    print(f"data: {len(dm.train)} train loans"
          + (f", {len(dm.val)} val loans" if dm.val is not None else ""), flush=True)

    model = CreditFoundationModel(
        vocab_size, n_field_types, dim=args.dim, n_heads=args.heads,
        profile_layers=args.profile_layers, event_layers=args.event_layers,
        history_layers=args.history_layers, dropout=args.dropout)
    print(f"model: {model.num_parameters()/1e6:.1f}M params (dim={args.dim}, dropout={args.dropout})",
          flush=True)

    history = train_mlm(
        model, dm, steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
        warmup=args.warmup, grad_clip=args.grad_clip, device=args.device, bf16=args.bf16,
        log_every=args.log_every, val_every=args.val_every)

    first, last = history["train"][0], history["train"][-1]
    msg = f"done: train loss {first:.4f} -> {last:.4f} over {args.steps} steps"
    if history["best_val"] is not None:
        msg += f"  | best val {history['best_val']:.4f} @ step {history['best_step']}"
    print(msg, flush=True)

    if args.out:
        ckpt = {
            "model": model.state_dict(),
            "config": {"vocab_size": vocab_size, "n_field_types": n_field_types, "dim": args.dim,
                       "n_heads": args.heads, "profile_layers": args.profile_layers,
                       "event_layers": args.event_layers, "history_layers": args.history_layers},
            "tokenizer": args.tokenizer, "steps": args.steps, "history": history,
        }
        storage.ensure_auth(args.out, args.key)
        with fsspec.open(args.out, "wb") as f:
            torch.save(ckpt, f)
        print(f"saved checkpoint -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
