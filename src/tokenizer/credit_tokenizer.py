"""Credit tokenizer (Phase 3 deliverable).

Converts borrower/loan/observation event sequences into token sequences for the
decoder foundation model. Token classes: categorical, numeric-bucket, temporal, event.

This is a scaffold — implement against the approved credit_event_schema and the
tokenizer config in configs/credit_tokenizer.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CreditTokenizerConfig:
    """Loaded from configs/credit_tokenizer.yaml."""
    numeric_buckets: dict = field(default_factory=dict)   # field -> bucket edges or "quantile:N"
    categorical_fields: list[str] = field(default_factory=list)
    temporal_fields: list[str] = field(default_factory=list)
    event_field: str = "event_type"
    max_context: int = 4096

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CreditTokenizerConfig":
        import yaml
        with open(path) as f:
            return cls(**yaml.safe_load(f))


class CreditTokenizer:
    def __init__(self, config: CreditTokenizerConfig):
        self.config = config
        self.vocab: dict[str, int] = {}

    def fit(self, sequences) -> "CreditTokenizer":
        """Build vocabulary from training event sequences."""
        raise NotImplementedError("Phase 3: build categorical/bucket/temporal/event vocab")

    def encode(self, sequence) -> list[int]:
        """Encode one event sequence into token ids."""
        raise NotImplementedError("Phase 3")

    def report(self) -> dict:
        """Token frequency, rare-token, OOV, and sequence-length stats (QA report)."""
        raise NotImplementedError("Phase 3: tokenizer QA report")
