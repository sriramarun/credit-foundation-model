# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Base model interface and shared transformer-encoder building blocks.
"""

from __future__ import annotations

import torch.nn as nn


class BaseEncoder(nn.Module):
    """Thin wrapper around a stack of transformer encoder layers."""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_layers, self.hidden_size, self.num_heads = num_layers, hidden_size, num_heads

    def forward(self, x, attn_mask=None):
        raise NotImplementedError
