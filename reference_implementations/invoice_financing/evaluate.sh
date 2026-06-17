#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Evaluation for the invoice_financing reference implementation.
set -euo pipefail
CFG=configs/invoice_financing
python scripts/extract_embeddings.py  --config $CFG/training.yaml
python scripts/evaluate_downstream.py --config $CFG/downstream_tasks.yaml
