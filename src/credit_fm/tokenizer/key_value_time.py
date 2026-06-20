# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Main key-value-time (KVT) tokenizer.

Decomposes each field into a semantic-type (key) token, value token(s), and a
temporal coordinate. encode() emits: [BOS] + origination block + per-cutoff event
blocks (delimited by [EVT_START]/[EVT_END]) + [EOS].
"""

from __future__ import annotations

from .base import BaseTokenizer


class KVTTokenizer(BaseTokenizer):
    def __init__(self, config: dict):
        self.config = config
        self.vocabulary = None  # built by build_vocab

    def build_vocab(self, train_panel) -> None:
        """Scan training data once to build the unified vocabulary."""
        raise NotImplementedError

    def encode(self, loan_panel) -> list[int]:
        raise NotImplementedError

    def decode(self, tokens: list[int]):
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> 'KVTTokenizer':
        raise NotImplementedError
