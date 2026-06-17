# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Three-source MLM masking: 15% individual tokens + 10% whole events + 10% semantic
types. A fraction of masked positions become [UNK] as input dropout.
"""

from __future__ import annotations


def apply_mlm_masking(tokens, token_rate=0.15, event_rate=0.10, type_rate=0.10):
    """Return (masked_tokens, mask_positions, targets)."""
    raise NotImplementedError
