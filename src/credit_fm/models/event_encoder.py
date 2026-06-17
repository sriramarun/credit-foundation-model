# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Event Encoder (4-5 layers default). Processes each cutoff's dynamic fields
independently; each event's [EVT] token output is its embedding. Calendar features
are added element-wise to the [EVT] embedding.
"""

from __future__ import annotations

from .base import BaseEncoder


class EventEncoder(BaseEncoder):
    def __init__(self, num_layers: int = 5, hidden_size: int = 256, num_heads: int = 8):
        super().__init__(num_layers, hidden_size, num_heads)

    def forward(self, event_tokens, calendar_features=None):
        raise NotImplementedError
