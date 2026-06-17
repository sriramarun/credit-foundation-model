# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Evaluation framework.
"""

from .metrics import roc_auc, pr_auc, ks_statistic, gini, brier
from .baselines import train_xgboost_baseline

__all__ = ['roc_auc','pr_auc','ks_statistic','gini','brier','train_xgboost_baseline']
