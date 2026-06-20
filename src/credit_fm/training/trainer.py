# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Main training loop. Default backend HuggingFace Trainer; NeMo AutoModel adapter
optional. Hyperparameters from training.yaml.
"""

from __future__ import annotations


class CreditTrainer:
    def __init__(self, model, config: dict, backend: str = 'hf'):
        assert backend in {'hf', 'nemo'}
        self.model, self.config, self.backend = model, config, backend

    def train(self, train_ds, val_ds):
        raise NotImplementedError
