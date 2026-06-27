# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Encode-once: turn a per-loan monthly panel into token-id **shards** for the data loader.

The model trains over each loan many times; re-tokenizing on every epoch would starve the GPUs.
So we encode every loan exactly once here (via :meth:`KVTTokenizer.encode_with_meta`) and persist
the result. Each row of a shard is one loan with four aligned ragged columns —
``input_ids`` / ``event_index`` / ``field_type`` / ``branch`` (the contract the hierarchical model
and the MLM masking both read) — plus ``n_tokens`` / ``n_events`` for batching and bucketing.

``encode_panel`` (one DataFrame → one shard DataFrame) is the testable core; ``iter_shards`` chunks
a panel into shard-sized DataFrames. The ``scripts/encode_dataset.py`` CLI wraps these with
pluggable storage so the same code writes local or ``gs://`` shards.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def _n_events(event_index: list[int]) -> int:
    """Number of event blocks = max month index + 1 (0 if the loan has no events)."""
    months = [e for e in event_index if e >= 0]
    return (max(months) + 1) if months else 0


def encode_panel(tokenizer, panel: pd.DataFrame) -> pd.DataFrame:
    """Encode every loan in ``panel`` to one shard DataFrame (one row per loan).

    Columns: ``<id_col>``, ``input_ids``, ``event_index``, ``field_type``, ``branch`` (ragged int
    lists), ``n_tokens``, ``n_events``.
    """
    idc = tokenizer.id_col
    records = []
    for loan_id, loan in panel.groupby(idc, sort=False):
        meta = tokenizer.encode_with_meta(loan)
        records.append({
            idc: loan_id,
            "input_ids": meta["input_ids"],
            "event_index": meta["event_index"],
            "field_type": meta["field_type"],
            "branch": meta["branch"],
            "n_tokens": len(meta["input_ids"]),
            "n_events": _n_events(meta["event_index"]),
        })
    return pd.DataFrame.from_records(records)


def iter_shards(tokenizer, panel: pd.DataFrame, shard_size: int) -> Iterator[pd.DataFrame]:
    """Yield encoded shard DataFrames of at most ``shard_size`` loans each.

    Loans are kept whole (grouped by id) and assigned to shards in first-seen order, so a loan's
    rows never split across shards.
    """
    idc = tokenizer.id_col
    loan_ids = panel[idc].drop_duplicates().tolist()
    for start in range(0, len(loan_ids), shard_size):
        chunk = set(loan_ids[start:start + shard_size])
        yield encode_panel(tokenizer, panel[panel[idc].isin(chunk)])
