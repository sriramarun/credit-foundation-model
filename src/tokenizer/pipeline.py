"""Generic tokenizer pipeline: ordered list of BaseTokenizer steps.

Mirrors NVIDIA's TokenizerPipeline. build_vocab() fits every step on the
training frame; tokenize() emits one interleaved token sequence per row,
ordered by entity and time.
"""
from .base import BaseTokenizer


class TokenizerPipeline:
    def __init__(self, steps: list[BaseTokenizer]):
        self.steps = steps

    def build_vocab(self, df) -> None:
        for step in self.steps:
            step.build_vocab(df)

    def tokenize(self, df):
        raise NotImplementedError("Phase 3: run steps and interleave per observation")

    @property
    def vocab(self) -> list[str]:
        out: list[str] = []
        for step in self.steps:
            out.extend(step.vocab)
        return out
