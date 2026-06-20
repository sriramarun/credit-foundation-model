# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""AdamW with linear warmup + cosine decay; optional Muon for larger models.
"""

from __future__ import annotations


def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.01, kind: str = 'adamw'):
    raise NotImplementedError


def build_scheduler(optimizer, warmup_steps: int = 500, total_steps: int = 10000):
    raise NotImplementedError
