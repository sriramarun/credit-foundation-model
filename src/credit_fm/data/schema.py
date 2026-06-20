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
