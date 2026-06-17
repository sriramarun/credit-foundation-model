# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""XGBoost/LightGBM baselines over engineered panel features (static + dynamic +
3m/12m lags), evaluated on the same test split as the foundation model.
"""

from __future__ import annotations


def train_xgboost_baseline(panel, task, split):
    raise NotImplementedError


def train_lightgbm_baseline(panel, task, split):
    raise NotImplementedError
