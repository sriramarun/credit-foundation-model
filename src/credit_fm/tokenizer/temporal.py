# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Temporal encoding: log-seconds since last event, 8*ln(1+seconds/8), plus cyclical
calendar features (hour/day-of-week/day-of-month) as (sin, cos) pairs.
"""

from __future__ import annotations

import math


def log_seconds(seconds: float) -> float:
    """Compress an inter-event gap in seconds."""
    return 8.0 * math.log(1.0 + seconds / 8.0)


def cyclical(value: float, period: float) -> tuple[float, float]:
    a = 2.0 * math.pi * value / period
    return math.sin(a), math.cos(a)


class TemporalEncoder:
    """Adds temporal coordinates to event-token embeddings."""
    def encode(self, timestamps):
        raise NotImplementedError
