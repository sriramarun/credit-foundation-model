# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""YAML config loader with light validation.
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
