#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Evaluation for the dutch_mortgages reference implementation.
set -euo pipefail
CFG=configs/dutch_mortgages
python scripts/extract_embeddings.py  --config $CFG/training.yaml
python scripts/evaluate_downstream.py --config $CFG/downstream_tasks.yaml
