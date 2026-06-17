# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""MLM pretraining head. For each masked position concatenates the Event-Encoder
output (local), History-Encoder output at the [EVT] position (cross-event), and
History-Encoder [USR] output (loan-level), then projects to vocab size.
"""

from __future__ import annotations

import torch.nn as nn


class MLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 3, vocab_size)

    def forward(self, local_ctx, event_ctx, user_ctx):
        raise NotImplementedError
