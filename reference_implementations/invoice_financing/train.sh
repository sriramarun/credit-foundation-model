#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# End-to-end training for the invoice_financing reference implementation.
set -euo pipefail
CFG=configs/invoice_financing
python scripts/prepare_data.py    --config $CFG/training.yaml
python scripts/train_baseline.py  --config $CFG/training.yaml
python scripts/train_tokenizer.py --config $CFG/tokenizer.yaml
python scripts/pretrain.py        --config $CFG/training.yaml --backend hf
