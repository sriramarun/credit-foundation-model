# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Checkpoint save/load helpers.
"""

from __future__ import annotations


def save_checkpoint(model, path: str, metadata: dict | None = None) -> None:
    raise NotImplementedError


def load_checkpoint(path: str):
    raise NotImplementedError
