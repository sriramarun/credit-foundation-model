# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Batch collation with sequence packing. Loans with similar event counts are batched
to minimize padding; flash-attn varlen prevents attention across loan boundaries.
"""

from __future__ import annotations


class PackedCollator:
    def __init__(self, pad_id: int = 0, max_length: int = 512):
        self.pad_id, self.max_length = pad_id, max_length

    def __call__(self, batch):
        raise NotImplementedError
