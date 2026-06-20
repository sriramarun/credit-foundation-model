# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""W&B logging, periodic checkpointing, and sanity-check sampling callbacks.
"""

from __future__ import annotations


class WandbLogger:
    def __init__(self, project: str = 'credit-foundation-model'):
        self.project = project

    def log(self, metrics: dict, step: int) -> None:
        raise NotImplementedError
