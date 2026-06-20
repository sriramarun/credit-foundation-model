# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Deterministic downstream label generators (unit-tested with edge cases).
"""

from __future__ import annotations


def default_within_k_months(panel, loan_id, observation_date, k: int = 6) -> int:
    raise NotImplementedError


def prepayment_within_k_months(panel, loan_id, observation_date, k: int = 6) -> int:
    raise NotImplementedError


def cure_within_k_months(panel, loan_id, observation_date, k: int = 3) -> int:
    raise NotImplementedError
