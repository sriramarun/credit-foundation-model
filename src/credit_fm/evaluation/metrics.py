# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Discriminative + calibration metrics. Each accepts (y_true, y_score).
"""

from __future__ import annotations


def roc_auc(y_true, y_score) -> float:
    raise NotImplementedError


def pr_auc(y_true, y_score) -> float:
    raise NotImplementedError


def ks_statistic(y_true, y_score) -> float:
    raise NotImplementedError


def gini(y_true, y_score) -> float:
    raise NotImplementedError


def brier(y_true, y_score) -> float:
    raise NotImplementedError


def lift_at_k(y_true, y_score, k: float = 0.05) -> float:
    raise NotImplementedError
