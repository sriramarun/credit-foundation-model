# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""LoRA fine-tuning wrapper (peft). Default rank=8, alpha=8 on QKV + MLP projections.
"""

from __future__ import annotations


def attach_lora(model, rank: int = 8, alpha: int = 8):
    """Return the model with LoRA adapters attached."""
    raise NotImplementedError
