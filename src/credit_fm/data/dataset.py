# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""PyTorch Dataset over a parquet panel + tokenizer. One loan -> one example:
{profile_tokens, event_tokens, labels}.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class CreditPanelDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer, labels=None):
        self.parquet_path, self.tokenizer, self.labels = parquet_path, tokenizer, labels

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
