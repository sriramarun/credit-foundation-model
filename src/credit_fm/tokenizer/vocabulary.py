# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Unified vocabulary across keys, values, and special tokens. JSON (de)serialization
and stats reporting (token frequency, sequence-length distribution, OOV rate).
"""

from __future__ import annotations

SPECIAL_TOKENS = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[UNK]',
                  '[USR]', '[EVT]', '[EVT_START]', '[EVT_END]']


class Vocabulary:
    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def add(self, token: str) -> int:
        raise NotImplementedError

    def to_json(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def from_json(cls, path) -> 'Vocabulary':
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError
