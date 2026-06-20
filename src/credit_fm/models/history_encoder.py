# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""History Encoder (4-6 layers default). Contextualizes [USR] + [EVT_1..EVT_n] with
time-to-last-event (RoPE). The [USR] output (z_h[0]) is the per-loan embedding.
"""

from __future__ import annotations

from .base import BaseEncoder


class HistoryEncoder(BaseEncoder):
    def __init__(self, num_layers: int = 6, hidden_size: int = 256, num_heads: int = 8):
        super().__init__(num_layers, hidden_size, num_heads)

    def forward(self, sequence, time_to_last=None):
        raise NotImplementedError
