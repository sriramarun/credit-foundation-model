# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Categorical value -> single token mapping. Reserves [UNK] for unseen values.
"""

from __future__ import annotations


class CategoricalTokenizer:
    UNK = '[UNK]'

    def __init__(self):
        self.value_to_id: dict = {}

    def fit(self, values) -> 'CategoricalTokenizer':
        raise NotImplementedError

    def transform(self, value) -> int:
        raise NotImplementedError
