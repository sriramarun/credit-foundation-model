# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Ablate the Profile State Encoder to reproduce the PRAGMA +31.8% PR-AUC finding.

Config-driven; pass a recipe with ``-c`` (not yet implemented).
"""

from credit_fm.utils.config import parse_cli
from credit_fm.utils.reproducibility import set_seed


def main() -> None:
    cfg = parse_cli(__doc__)
    set_seed(cfg.get_path("seed", 42))
    raise NotImplementedError


if __name__ == '__main__':
    main()
