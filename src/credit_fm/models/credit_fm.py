# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Full model composing Profile State, Event, and History encoders.
"""

from __future__ import annotations

import torch.nn as nn

from .profile_encoder import ProfileStateEncoder
from .event_encoder import EventEncoder
from .history_encoder import HistoryEncoder
from .mlm_head import MLMHead


class CreditFoundationModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.profile_encoder = ProfileStateEncoder()
        self.event_encoder = EventEncoder()
        self.history_encoder = HistoryEncoder()
        self.mlm_head = None  # built once vocab_size is known

    def forward(self, profile_tokens, event_tokens, masked_positions):
        """Return the MLM loss for pretraining."""
        raise NotImplementedError

    def extract_embeddings(self, profile_tokens, event_tokens):
        """Return the per-loan [USR] embedding for inference."""
        raise NotImplementedError
