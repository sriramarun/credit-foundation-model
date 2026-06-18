#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Idempotent environment bring-up for the NGC PyTorch H100 container.
# Creates a restart-proof venv under /workspace, installs the credit_fm package,
# wires auto-activation + secrets, and verifies the scaffold.
#
# Usage:  bash scripts/setup_container.sh
# Safe to re-run after a container restart or recreate.
#
# See docs/container_setup.md for the full explanation.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="$WORKSPACE/.venv"
REPO="${REPO:-$WORKSPACE/credit-foundation-model}"
SECRETS="$WORKSPACE/secrets.env"

echo "==> 1. GPU / torch check (image build)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

echo "==> 2. Create venv (inherits image's CUDA torch via --system-site-packages)"
if [ ! -d "$VENV" ]; then
  python -m venv "$VENV" --system-site-packages
  echo "    created $VENV"
else
  echo "    $VENV already exists — reusing"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "==> 3. Install credit_fm (editable, dev extras; do not reinstall torch)"
pip install -e "${REPO}[dev]" --no-build-isolation
python -c "import torch; print('    torch still', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "==> 4. Wire auto-activation (grep-guarded, survives restarts)"
grep -q "$VENV/bin/activate" ~/.bashrc 2>/dev/null \
  || echo "source $VENV/bin/activate" >> ~/.bashrc

echo "==> 5. Secrets (sourced if present; never commit this file)"
if [ -f "$SECRETS" ]; then
  grep -q "$SECRETS" ~/.bashrc 2>/dev/null || echo "source $SECRETS" >> ~/.bashrc
  # shellcheck disable=SC1090
  source "$SECRETS"
  echo "    sourced $SECRETS"
else
  echo "    no $SECRETS found — create it with WANDB_API_KEY / HF_TOKEN (see docs)"
fi

echo "==> 6. Verify scaffold"
python -c "import credit_fm; print('    credit_fm', credit_fm.__version__)"
( cd "$REPO" && ruff check . && pytest -q ) || echo "    (ruff/pytest reported issues — review above)"

echo "==> Done. Open a new shell or 'source $VENV/bin/activate' to use the env."
