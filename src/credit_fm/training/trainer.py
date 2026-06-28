# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""MLM training loop for the credit foundation model.

``train_mlm`` is the reusable core (single-GPU/CPU; bf16 autocast on CUDA) used by the M2 toy run
and the M3 pretraining script. It pulls batches from a :class:`CreditDataModule`, optimises the MLM
loss with AdamW + warmup-cosine, clips gradients, and periodically reports validation loss. It
returns a history dict for logging/plotting.

``CreditTrainer`` is a thin object wrapper kept for the scaffold interface; the HF-Trainer / NeMo
backends are deferred to M3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

from .optimizers import build_optimizer, build_scheduler


def _cycle(loader) -> Iterator:
    while True:
        yield from loader


def _to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def _evaluate(model, datamodule, device: str, use_amp: bool) -> float:
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    for batch in datamodule.val_dataloader():
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            total += model(batch)["loss"].item()
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def train_mlm(model, datamodule, *, steps: int = 100, lr: float = 3e-4, weight_decay: float = 0.01,
              warmup: int = 10, grad_clip: float = 1.0, min_lr_ratio: float = 0.1,
              device: str | None = None, bf16: bool = False, log_every: int = 10,
              val_every: int = 0, log: Callable[[str], None] = print) -> dict:
    """Train ``model`` for ``steps`` optimiser steps on ``datamodule``; return a loss history."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bf16 and device.startswith("cuda")
    model.to(device).train()
    opt = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    sched = build_scheduler(opt, warmup_steps=warmup, total_steps=steps, min_lr_ratio=min_lr_ratio)

    history: dict = {"train": [], "val": []}
    batches = _cycle(datamodule.train_dataloader())
    for step in range(1, steps + 1):
        batch = _to_device(next(batches), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = model(batch)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        history["train"].append(loss.item())
        if log_every and (step % log_every == 0 or step == 1):
            log(f"step {step}/{steps}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}")
        if val_every and datamodule.val is not None and step % val_every == 0:
            v = _evaluate(model, datamodule, device, use_amp)
            history["val"].append((step, v))
            log(f"  [val] step {step}  loss {v:.4f}")
    return history


class CreditTrainer:
    """Scaffold wrapper around :func:`train_mlm` (HF/NeMo backends deferred to M3)."""

    def __init__(self, model, config: dict, backend: str = "hf"):
        assert backend in {"hf", "nemo"}
        self.model, self.config, self.backend = model, config, backend

    def train(self, datamodule) -> dict:
        return train_mlm(self.model, datamodule, **self.config)
