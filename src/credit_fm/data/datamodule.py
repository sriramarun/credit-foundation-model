# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Optional Lightning DataModule wrapping train/val/test datasets.
"""

from __future__ import annotations


class CreditDataModule:
    def __init__(self, config: dict):
        self.config = config

    def setup(self, stage: str | None = None):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError
