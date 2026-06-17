# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Convert raw asset-class data into the canonical credit panel parquet. Reuses the
53-test SQL validation suite for schema/domain/lifecycle checks.
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
