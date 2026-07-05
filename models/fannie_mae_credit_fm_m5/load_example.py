# SPDX-License-Identifier: Apache-2.0
"""Load this published credit foundation model and extract a per-loan embedding."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from credit_fm.models import CreditFoundationModel  # pip install -e . from the credit_fm repo

here = Path(__file__).parent
cfg = json.loads((here / "config.json").read_text())
model = CreditFoundationModel(
    cfg["vocab_size"], cfg["n_field_types"], dim=cfg["dim"], n_heads=cfg["n_heads"],
    profile_layers=cfg["profile_layers"], event_layers=cfg["event_layers"],
    history_layers=cfg["history_layers"])
model.load_state_dict(load_file(here / "model.safetensors"))
model.eval()
print(f"loaded {model.num_parameters()/1e6:.1f}M-param credit FM; encode loans with tokenizer.json")
