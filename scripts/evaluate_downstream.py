# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Run the three-way comparison (baseline / embeddings-only / combined / LoRA).
"""

import argparse

from credit_fm.utils.reproducibility import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    raise NotImplementedError


if __name__ == '__main__':
    main()
