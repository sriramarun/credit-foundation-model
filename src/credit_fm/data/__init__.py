# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Data layer: schema, dataset, splits, labels.
"""

from .schema import CreditPanelSchema
from .dataset import CreditPanelDataset
from .encode import encode_panel, iter_shards
from .splits import temporal_loan_split

__all__ = ["CreditPanelSchema", "CreditPanelDataset", "encode_panel",
           "iter_shards", "temporal_loan_split"]
