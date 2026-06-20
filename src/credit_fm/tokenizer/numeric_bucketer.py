# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Percentile bucketing for continuous fields. Reserves a separate bucket for zero;
bin edges are fit on training data only (never leak from test).
"""

from __future__ import annotations


class NumericBucketer:
    def __init__(self, n_bins: int = 16):
        self.n_bins = n_bins
        self.edges = None

    def fit(self, values) -> 'NumericBucketer':
        """Compute quantile bin edges from training values."""
        raise NotImplementedError

    def transform(self, value) -> int:
        """Return the bucket index for a value."""
        raise NotImplementedError
