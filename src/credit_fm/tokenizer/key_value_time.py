# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Main key-value-time (KVT) tokenizer.

Composes the per-field tokenizers (numeric bucketing, categorical) into one loan-level encoder.
Fields are tagged to a **branch**: *profile* (static origination facts, emitted once) and *event*
(per-month dynamic facts, emitted per monthly row). Tokens are **fused** ``field=value`` strings
(NVIDIA-TFM style) mapped to ids via the shared :class:`Vocabulary`.

A loan encodes to::

    [BOS] [USR]
      <profile tokens: original_ltv=4, channel=R, ...>           # once, from the loan's first row
      [EVT_START] t=<age_bin> <event tokens: current_interest_rate=7, ...> [EVT_END]   # per month
      ...
    [EOS]

Everything is fit on TRAIN only and serialized (vocab + bin edges + categories), so val / test /
inference reuse identical ids. Config (per asset) drives which fields go to which branch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseTokenizer
from .categorical import CategoricalTokenizer
from .numeric_bucketer import NumericBucketer
from .vocabulary import Vocabulary


class KVTTokenizer(BaseTokenizer):
    """Config-driven KVT tokenizer.

    config keys::
        id_col, time_col, time_field
        profile: {numeric: [...], categorical: [...]}
        event:   {numeric: [...], categorical: [...]}
        n_bins (16), max_categories (256), max_events (60)
    """

    def __init__(self, config: dict):
        self.config = config
        self.vocabulary: Vocabulary | None = None
        self._num: dict[str, NumericBucketer] = {}
        self._cat: dict[str, CategoricalTokenizer] = {}
        self._time: NumericBucketer | None = None
        self._parse_config()

    def _parse_config(self) -> None:
        c = self.config
        self.id_col = c["id_col"]
        self.time_col = c["time_col"]
        self.time_field = c["time_field"]
        self.p_num = list(c.get("profile", {}).get("numeric", []))
        self.p_cat = list(c.get("profile", {}).get("categorical", []))
        self.e_num = list(c.get("event", {}).get("numeric", []))
        self.e_cat = list(c.get("event", {}).get("categorical", []))
        self.n_bins = int(c.get("n_bins", 16))
        self.max_categories = int(c.get("max_categories", 256))
        self.max_events = int(c.get("max_events", 60))

    # ------------------------------------------------------------------ fit
    def fit(self, panel: pd.DataFrame) -> 'KVTTokenizer':
        """Fit every field tokenizer on the training panel and build the vocabulary."""
        self._num = {f: NumericBucketer(self.n_bins).fit(panel[f]) for f in self.p_num + self.e_num}
        self._cat = {f: CategoricalTokenizer(self.max_categories).fit(panel[f])
                     for f in self.p_cat + self.e_cat}
        self._time = NumericBucketer(self.n_bins).fit(panel[self.time_field])

        vocab = Vocabulary()
        for f in self.p_num + self.e_num:
            for label in self._num[f].vocab():
                vocab.add(f"{f}={label}")
        for f in self.p_cat + self.e_cat:
            for label in self._cat[f].vocab():
                vocab.add(f"{f}={label}")
        for label in self._time.vocab():
            vocab.add(f"t={label}")
        self.vocabulary = vocab
        return self

    def build_vocab(self, train_panel: pd.DataFrame) -> None:
        """Alias for fit() (scaffold-compatible)."""
        self.fit(train_panel)

    # --------------------------------------------------------------- encode
    def _profile_tokens(self, row: pd.Series) -> list[str]:
        toks = [f"{f}={self._num[f].transform(row.get(f))}" for f in self.p_num]
        toks += [f"{f}={self._cat[f].transform(row.get(f))}" for f in self.p_cat]
        return toks

    def _event_tokens(self, row: pd.Series) -> list[str]:
        toks = ["[EVT_START]", f"t={self._time.transform(row.get(self.time_field))}"]
        toks += [f"{f}={self._num[f].transform(row.get(f))}" for f in self.e_num]
        toks += [f"{f}={self._cat[f].transform(row.get(f))}" for f in self.e_cat]
        toks.append("[EVT_END]")
        return toks

    def tokens(self, loan_panel: pd.DataFrame) -> list[str]:
        """Encode one loan's rows into the fused token *strings* (pre-id, for QA/inspection)."""
        if self.vocabulary is None:
            raise RuntimeError("tokenizer not fitted — call fit(train_panel) first")
        df = loan_panel.sort_values(self.time_col).tail(self.max_events)
        seq = ["[BOS]", "[USR]"]
        seq += self._profile_tokens(df.iloc[0])
        for _, row in df.iterrows():
            seq += self._event_tokens(row)
        seq.append("[EOS]")
        return seq

    def encode(self, loan_panel: pd.DataFrame) -> list[int]:
        return [self.vocabulary.encode(t) for t in self.tokens(loan_panel)]

    def decode(self, tokens: list[int]) -> list[str]:
        return [self.vocabulary.decode(i) for i in tokens]

    @property
    def vocab_size(self) -> int:
        return self.vocabulary.size if self.vocabulary is not None else 0

    # ------------------------------------------------------------ serialize
    @staticmethod
    def _num_state(nb: NumericBucketer) -> dict:
        return {"n_bins": nb.n_bins, "n_bins_": nb.n_bins_,
                "edges": nb.edges.tolist() if nb.edges is not None else None}

    @staticmethod
    def _num_from(state: dict) -> NumericBucketer:
        nb = NumericBucketer(state["n_bins"])
        nb.n_bins_ = state["n_bins_"]
        nb.edges = np.array(state["edges"]) if state["edges"] is not None else None
        return nb

    def save(self, path) -> None:
        state = {
            "config": self.config,
            "numeric": {f: self._num_state(nb) for f, nb in self._num.items()},
            "categorical": {f: {"max_categories": ct.max_categories, "min_count": ct.min_count,
                                "categories_": ct.categories_} for f, ct in self._cat.items()},
            "time": self._num_state(self._time),
            "vocab": [self.vocabulary.id_to_token[i] for i in range(self.vocabulary.size)],
        }
        Path(path).write_text(json.dumps(state))

    @classmethod
    def load(cls, path) -> 'KVTTokenizer':
        state = json.loads(Path(path).read_text())
        obj = cls(state["config"])
        obj._num = {f: cls._num_from(s) for f, s in state["numeric"].items()}
        obj._cat = {}
        for f, s in state["categorical"].items():
            ct = CategoricalTokenizer(s["max_categories"], s["min_count"])
            ct.categories_ = s["categories_"]
            obj._cat[f] = ct
        obj._time = cls._num_from(state["time"])
        vocab = Vocabulary.__new__(Vocabulary)
        vocab.token_to_id = {t: i for i, t in enumerate(state["vocab"])}
        vocab.id_to_token = {i: t for i, t in enumerate(state["vocab"])}
        obj.vocabulary = vocab
        return obj
