"""Hash-based categorical tokenizer for high-cardinality credit fields.

Maps high-cardinality strings (e.g. servicer, MSA, originator) to a fixed
number of hash buckets via modulo. Caller pre-hashes the column to ints.
"""
from .base import BaseTokenizer


class CategoricalHashTokenizer(BaseTokenizer):
    def __init__(self, column: str, prefix: str, hash_size: int):
        self.column, self.prefix, self.hash_size = column, prefix, hash_size

    def build_vocab(self, df) -> None:
        pass

    def tokenize(self, df):
        raise NotImplementedError("Phase 3")

    @property
    def vocab(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.hash_size)]
