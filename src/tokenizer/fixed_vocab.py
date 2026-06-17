"""Fixed-vocabulary tokenizer for bounded integer credit fields.

Maps integers from a known range to token strings like "MOB_012"
(months-on-book), "DPD_BUCKET_2", "TERM_360". No data-driven fitting.
"""
from .base import BaseTokenizer


class FixedVocabTokenizer(BaseTokenizer):
    def __init__(self, column: str, prefix: str, min_val: int, max_val: int, pad_width: int = 0):
        self.column, self.prefix = column, prefix
        self.min_val, self.max_val, self.pad_width = min_val, max_val, pad_width

    def build_vocab(self, df) -> None:  # no fitting needed
        pass

    def tokenize(self, df):
        raise NotImplementedError("Phase 3")

    @property
    def vocab(self) -> list[str]:
        w = self.pad_width
        return [f"{self.prefix}_{v:0{w}d}" for v in range(self.min_val, self.max_val + 1)]
