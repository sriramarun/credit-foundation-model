"""Numeric binning tokenizer for continuous credit fields.

Bins continuous values (balance, interest rate, LTV, DTI, FICO) into tokens
like "BAL_7". Supports fixed edges or data-driven quantile/optimal binning.
"""
from .base import BaseTokenizer


class NumericalTokenizerOptBin(BaseTokenizer):
    def __init__(self, column: str, prefix: str, n_bins: int = 32, strategy: str = "quantile"):
        self.column, self.prefix = column, prefix
        self.n_bins, self.strategy = n_bins, strategy
        self._edges = None

    def build_vocab(self, df) -> None:
        raise NotImplementedError("Phase 3: fit bin edges (quantile/optbin)")

    def tokenize(self, df):
        raise NotImplementedError("Phase 3")

    @property
    def vocab(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.n_bins)]
