# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Split tests: loan-level disjointness, completeness, and temporal ordering."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_fm.data.splits import temporal_loan_split


def _toy_origination(n_loans: int = 100) -> pd.Series:
    """n loans with strictly increasing origination dates, indexed by loan_id."""
    return pd.Series(
        pd.date_range("2020-01-01", periods=n_loans, freq="D"),
        index=[f"L{i:03d}" for i in range(n_loans)],
        name="origination",
    ).rename_axis("loan_id")


def test_split_is_disjoint_and_complete():
    orig = _toy_origination(100)
    assignment = temporal_loan_split(orig, fractions=(0.8, 0.1, 0.1))

    assert set(assignment) == set(orig.index)

    by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for lid, s in assignment.items():
        by_split[s].append(lid)

    assert set(by_split["train"]).isdisjoint(by_split["val"])
    assert set(by_split["val"]).isdisjoint(by_split["test"])
    assert set(by_split["train"]).isdisjoint(by_split["test"])

    assert (len(by_split["train"]), len(by_split["val"]), len(by_split["test"])) == (80, 10, 10)


def test_split_is_temporal():
    orig = _toy_origination(100)
    s = pd.Series(temporal_loan_split(orig))
    train_max = orig[s[s == "train"].index].max()
    val_min = orig[s[s == "val"].index].min()
    val_max = orig[s[s == "val"].index].max()
    test_min = orig[s[s == "test"].index].min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        temporal_loan_split(_toy_origination(10), fractions=(0.7, 0.1, 0.1))


def test_rejects_missing_origination():
    orig = _toy_origination(10)
    orig.iloc[3] = pd.NaT
    with pytest.raises(ValueError):
        temporal_loan_split(orig)
