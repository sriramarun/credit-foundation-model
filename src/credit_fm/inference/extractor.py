# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Load a pretrained checkpoint and produce per-loan embeddings, saved as parquet.
"""

from __future__ import annotations


class EmbeddingExtractor:
    def __init__(self, checkpoint_dir: str, pooling: str = 'usr'):
        self.checkpoint_dir, self.pooling = checkpoint_dir, pooling

    def extract(self, dataset, out_path: str) -> None:
        raise NotImplementedError
