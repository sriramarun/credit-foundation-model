"""Inter-event time-delta tokenizer.

Log-compresses time between consecutive observations/payments into tokens
like "DT_4", capturing payment cadence and gaps in credit histories.
"""
from .base import BaseTokenizer


class TimeDeltaTokenizer(BaseTokenizer):
    def __init__(self, time_column: str, entity_column: str, prefix: str = "DT", n_bins: int = 16):
        self.time_column, self.entity_column = time_column, entity_column
        self.prefix, self.n_bins = prefix, n_bins

    def build_vocab(self, df) -> None:
        raise NotImplementedError("Phase 3: fit log-time-delta bins")

    def tokenize(self, df):
        raise NotImplementedError("Phase 3")

    @property
    def vocab(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(self.n_bins)]
