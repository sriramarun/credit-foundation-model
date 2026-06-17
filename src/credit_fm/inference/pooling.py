# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Pooling strategies: last-token, mean, attention-weighted, [USR]-token (default),
final-[EVT]-token.
"""

from __future__ import annotations

POOLING_STRATEGIES = ['last', 'mean', 'attention', 'usr', 'last_evt']


def pool(hidden_states, strategy: str = 'usr', mask=None):
    assert strategy in POOLING_STRATEGIES
    raise NotImplementedError
