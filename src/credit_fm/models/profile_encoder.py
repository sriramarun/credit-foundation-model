# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Profile State Encoder (3 layers default). Processes static fields plus life-long
events with RoPE timestamps; the [USR] position is the aggregated profile embedding.
"""

from __future__ import annotations

from .base import BaseEncoder


class ProfileStateEncoder(BaseEncoder):
    def __init__(self, num_layers: int = 3, hidden_size: int = 256, num_heads: int = 8):
        super().__init__(num_layers, hidden_size, num_heads)

    def forward(self, profile_tokens, timestamps=None):
        raise NotImplementedError
