# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Downstream task definitions: default_6m, default_3m, prepayment_6m, cure_3m,
segmentation. Each binds a label generator, splits, and a primary metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Benchmark:
    name: str
    label_fn: Callable
    primary_metric: str


BENCHMARKS: dict[str, Benchmark] = {}
