# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Pretrain the credit foundation model (MLM) over encode-once shards.

Loads a frozen ``tokenizer.json`` (for ``vocab_size`` + ``n_field_types``), builds a
``CreditDataModule`` over the encoded shard directories, constructs ``CreditFoundationModel``, and
runs ``train_mlm``. The checkpoint stores the model config (for reload) and the full resolved
run config (lineage).

Config-driven (recipe: ``configs/fannie_mae/pretrain.yaml``)::

    python scripts/pretrain.py -c configs/fannie_mae/pretrain.yaml                # full run
    python scripts/pretrain.py -c configs/fannie_mae/pretrain.yaml \
        --data.limit 1000 --schedule.steps 100 --model.dim 256 \
        --checkpoint.out runs/toy.pt                                              # toy run
"""

from __future__ import annotations

import os

# reduce CUDA fragmentation for long-sequence O(L²) attention — must be set BEFORE torch inits CUDA.
# (FlashAttention/SDPA already avoids the big score matrix; this is belt-and-suspenders for headroom.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import fsspec  # noqa: E402
import torch  # noqa: E402

from credit_fm.data import CreditDataModule  # noqa: E402
from credit_fm.models import CreditFoundationModel  # noqa: E402
from credit_fm.tokenizer import KVTTokenizer  # noqa: E402
from credit_fm.training import train_mlm  # noqa: E402
from credit_fm.utils import storage  # noqa: E402
from credit_fm.utils.config import parse_cli, summarize  # noqa: E402
from credit_fm.utils.reproducibility import set_seed  # noqa: E402


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/fannie_mae/pretrain.yaml")
    print(f"config: {cfg.config_path}\n"
          f"{summarize(cfg, 'data', 'model', 'optimizer', 'schedule', 'runtime', 'checkpoint')}",
          flush=True)
    set_seed(cfg.seed)

    tok = KVTTokenizer.load(cfg.tokenizer)
    vocab_size, n_field_types = tok.vocab_size, len(tok.field_types)
    print(f"tokenizer: {vocab_size} tokens, {n_field_types} field types", flush=True)

    dm = CreditDataModule(cfg.data.train_dir, val_dir=cfg.data.val_dir, vocab_size=vocab_size,
                          batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
                          limit=cfg.data.limit, key=cfg.key)
    print(f"data: {len(dm.train)} train loans"
          + (f", {len(dm.val)} val loans" if dm.val is not None else ""), flush=True)

    m = cfg.model
    model = CreditFoundationModel(
        vocab_size, n_field_types, dim=m.dim, n_heads=m.n_heads,
        profile_layers=m.profile_layers, event_layers=m.event_layers,
        history_layers=m.history_layers, dropout=m.dropout)
    print(f"model: {model.num_parameters()/1e6:.1f}M params (dim={m.dim}, dropout={m.dropout})",
          flush=True)

    grad_accum = int(cfg.get_path("schedule.grad_accum") or 1)
    if grad_accum > 1:
        print(f"grad accumulation: {grad_accum} micro-batches/step "
              f"(effective batch {cfg.data.batch_size * grad_accum})", flush=True)
    ckpt_every = int(cfg.get_path("checkpoint.every") or 0)
    if ckpt_every:
        print(f"step checkpoints: every {ckpt_every} steps, keep {cfg.get_path('checkpoint.keep', 2)}"
              f" (resume with --resume auto)", flush=True)
    history = train_mlm(
        model, dm, steps=cfg.schedule.steps, lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay, warmup=cfg.optimizer.warmup,
        grad_clip=cfg.optimizer.grad_clip, grad_accum=grad_accum,
        device=cfg.runtime.device, bf16=cfg.runtime.bf16,
        log_every=cfg.schedule.log_every, val_every=cfg.schedule.val_every,
        checkpoint_every=ckpt_every, checkpoint_keep=int(cfg.get_path("checkpoint.keep", 2) or 2),
        checkpoint_out=cfg.get_path("checkpoint.out"), resume=cfg.get_path("resume"))

    first, last = history["train"][0], history["train"][-1]
    msg = f"done: train loss {first:.4f} -> {last:.4f} over {cfg.schedule.steps} steps"
    if history["best_val"] is not None:
        msg += f"  | best val {history['best_val']:.4f} @ step {history['best_step']}"
    print(msg, flush=True)

    out = cfg.get_path("checkpoint.out")
    if out:
        ckpt = {
            "model": model.state_dict(),
            "config": {"vocab_size": vocab_size, "n_field_types": n_field_types, "dim": m.dim,
                       "n_heads": m.n_heads, "profile_layers": m.profile_layers,
                       "event_layers": m.event_layers, "history_layers": m.history_layers},
            "run_config": cfg.to_dict(),                           # lineage
            "tokenizer": cfg.tokenizer, "steps": cfg.schedule.steps, "history": history,
        }
        storage.ensure_auth(out, cfg.key)
        with fsspec.open(out, "wb") as f:
            torch.save(ckpt, f)
        print(f"saved checkpoint -> {out}", flush=True)


if __name__ == "__main__":
    main()
