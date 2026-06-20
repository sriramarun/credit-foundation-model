# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Canonical credit panel schema: one row per (loan_id, observation_date). Required
columns: loan_id, observation_date, is_origination, event_type; plus asset-class
fields. Static fields repeat across observations; dynamic fields vary.
"""

from __future__ import annotations

from dataclasses import dataclass, field


REQUIRED_COLUMNS = ['loan_id', 'observation_date', 'is_origination', 'event_type']


@dataclass
class CreditPanelSchema:
    static_fields: list[str] = field(default_factory=list)
    dynamic_fields: list[str] = field(default_factory=list)
    categorical_fields: list[str] = field(default_factory=list)
    numeric_fields: list[str] = field(default_factory=list)

    def validate(self, df) -> None:
        """Assert the dataframe satisfies the canonical schema."""
        raise NotImplementedError


def classify_fields(
    df,
    id_col: str = "loan_id",
    time_col: str = "reporting_date",
    num_unique_cat: int = 20,
    sample_loans: int = 20000,
    seed: int = 42,
) -> dict:
    """Classify every column by role and value type for tokenizer config.

    role  : ``id`` | ``static`` (constant within a loan) | ``dynamic`` (varies per cutoff).
    type  : ``constant`` (1 value → drop) | ``temporal`` | ``flag`` | ``bucket`` |
            ``numeric`` | ``categorical`` | ``text``.

    Static/dynamic is decided on a random sample of loans for speed (the property is
    structural, so a sample is representative). Run on ``train.parquet`` so any downstream
    cardinality stats stay train-only.

    Returns ``{col: {"role", "type", "n_unique", "null_frac"}}``.
    """
    import pandas as pd

    loans = pd.Series(df[id_col].unique())
    if sample_loans and len(loans) > sample_loans:
        keep = set(loans.sample(sample_loans, random_state=seed))
        grouped = df[df[id_col].isin(keep)].groupby(id_col)
    else:
        grouped = df.groupby(id_col)

    out: dict[str, dict] = {}
    for c in df.columns:
        col = df[c]
        name = c.lower()
        n_unique = int(col.nunique(dropna=True))
        if c == id_col:
            role, ftype = "id", "text"
        else:
            role = "static" if grouped[c].nunique(dropna=False).max() <= 1 else "dynamic"
            if n_unique <= 1:
                ftype = "constant"
            elif c == time_col or name.endswith("_date") or "maturity_date" in name:
                ftype = "temporal"
            elif name.endswith("_flag"):
                ftype = "flag"
            elif name.endswith("_bucket"):
                ftype = "bucket"
            elif pd.api.types.is_numeric_dtype(col):
                ftype = "numeric" if n_unique > num_unique_cat else "categorical"
            else:
                avg_len = col.dropna().astype(str).str.len().mean()
                ftype = "text" if (n_unique > 1000 or avg_len > 40) else "categorical"
        out[c] = {
            "role": role,
            "type": ftype,
            "n_unique": n_unique,
            "null_frac": round(float(col.isna().mean()), 4),
        }
    return out
