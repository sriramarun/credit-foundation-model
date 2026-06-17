"""Mapping-based categorical tokenizer for low-cardinality credit fields.

E.g. product_type, state, delinquency_status, lien_position. Builds a
value->token map from the training frame.
"""
from .base import BaseTokenizer


class MappingTokenizer(BaseTokenizer):
    def __init__(self, column: str, prefix: str):
        self.column, self.prefix = column, prefix
        self._map: dict = {}

    def build_vocab(self, df) -> None:
        raise NotImplementedError("Phase 3: build value->token map from training data")

    def tokenize(self, df):
        raise NotImplementedError("Phase 3")

    @property
    def vocab(self) -> list[str]:
        return list(self._map.values())
