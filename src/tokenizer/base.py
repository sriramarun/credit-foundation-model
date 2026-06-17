"""Abstract base class for modular tokenizer steps.

Mirrors the NVIDIA TFM tokenizer design: each step handles one or more
DataFrame columns and converts raw values into token strings (e.g.
"BAL_3", "DPD_30"). Configuration lives in __init__; data flows only
through build_vocab() and tokenize().
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """One tokenizer step over one or more columns."""

    @abstractmethod
    def build_vocab(self, df) -> None:
        """Fit any data-driven state (bins, mappings) from a training frame."""

    @abstractmethod
    def tokenize(self, df):
        """Return token strings for the configured column(s)."""

    @property
    @abstractmethod
    def vocab(self) -> list[str]:
        """All token strings this step can emit."""
