# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Package a pretrained checkpoint into a self-contained, publishable model directory.

Reads a ``.pt`` checkpoint (local or ``gs://``) and writes a release folder containing:

* ``model.safetensors`` — the weights in the safetensors format (no pickle; loads anywhere),
* ``config.json``       — the architecture config needed to rebuild ``CreditFoundationModel``,
* ``tokenizer.json``    — a copy of the frozen KVT tokenizer (required to encode inputs),
* ``README.md``         — the model card (copied from ``--model-card``),
* ``load_example.py``   — a minimal snippet that reconstructs the model and loads the weights,
* ``LICENSE``           — Apache-2.0 (copied from the repo root if present).

The folder is then uploaded to Hugging Face Hub (``huggingface-cli upload``) or committed via
Git LFS — see the printed next-steps. Blueprint parity: the NVIDIA transaction-FM example ships
HF-loadable safetensors; ours needs the ``credit_fm`` package to instantiate the custom
three-branch architecture, so we ship weights + config + a load example rather than an
``AutoModel`` shim.

    python scripts/publish_model.py -c configs/mortgage_performance/publish.yaml
    python scripts/publish_model.py -c configs/mortgage_performance/publish.yaml --out models/mortgage_performance_m5
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import fsspec
import torch
from safetensors.torch import save_file

from credit_fm.utils import storage
from credit_fm.utils.config import parse_cli, summarize

_LOAD_EXAMPLE = '''# SPDX-License-Identifier: Apache-2.0
"""Runnable example: load the published credit foundation model and use it.

    pip install "credit_fm @ git+https://github.com/Algoritmica-ai/deeploans.git#subdirectory=credit-foundation-model"
    python load_example.py

Covers both artifacts in this repository:
  * the pretrained backbone (repo root)      -> per-loan embeddings
  * the fine-tuned head (finetuned-default-12m/) -> a 12-month default score

THE INPUT FORMAT IS THE PART WORTH READING. The model does not take text or a
feature row: it takes an already-tokenized loan panel as a dict of five aligned
tensors. In production these come from the framework's encoding pipeline
(`scripts/encode_dataset.py`); the synthetic batch below exists so you can verify
the weights load and run before wiring up real data.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from credit_fm.models import CreditFoundationModel

HERE = Path(__file__).parent


def load(directory: Path) -> tuple[CreditFoundationModel, dict]:
    """Rebuild the architecture from config.json and load its weights."""
    cfg = json.loads((directory / "config.json").read_text())
    model = CreditFoundationModel(
        cfg["vocab_size"], cfg["n_field_types"], dim=cfg["dim"], n_heads=cfg["n_heads"],
        profile_layers=cfg["profile_layers"], event_layers=cfg["event_layers"],
        history_layers=cfg["history_layers"])
    model.load_state_dict(load_file(directory / "model.safetensors"), strict=True)
    model.eval()
    return model, cfg


def example_batch(cfg: dict, n_loans: int = 2, n_tokens: int = 48, n_events: int = 6) -> dict:
    """A synthetic batch in the exact shape the model expects.

    Every tensor is ``(n_loans, n_tokens)`` and positionally aligned — entry *i* of each
    describes the same token:

    ``input_ids``    token id from ``tokenizer.json`` (a fused ``field=value`` token).
    ``field_type``   which field the token belongs to; lets the model tell a rate from a balance.
    ``branch``       0 = profile token (static, fixed at origination), 1 = event token (monthly).
    ``event_index``  which month the token belongs to (0-based); -1 for profile tokens.
    ``n_events``     per loan, how many monthly events it actually has — shape ``(n_loans,)``.

    Real batches carry the same keys with real ids; the layout below (a profile block followed
    by monthly event blocks) mirrors how a loan is serialised.
    """
    torch.manual_seed(0)
    n_profile = 12
    return {
        "input_ids": torch.randint(0, cfg["vocab_size"], (n_loans, n_tokens)),
        "field_type": torch.randint(0, cfg["n_field_types"], (n_loans, n_tokens)),
        "branch": torch.cat([torch.zeros(n_loans, n_profile, dtype=torch.long),
                             torch.ones(n_loans, n_tokens - n_profile, dtype=torch.long)], dim=1),
        "event_index": torch.cat([torch.full((n_loans, n_profile), -1, dtype=torch.long),
                                  torch.arange(n_tokens - n_profile).remainder(n_events)
                                       .expand(n_loans, -1)], dim=1),
        "n_events": torch.full((n_loans,), n_events, dtype=torch.long),
    }


def main() -> None:
    # ---- 1. the backbone: a loan history -> one vector -------------------------------
    model, cfg = load(HERE)
    batch = example_batch(cfg)
    with torch.no_grad():
        out = model(batch)
    emb = out["loan_embedding"]
    print(f"backbone: {cfg['n_parameters'] / 1e6:.1f}M params")
    print(f"  loan_embedding {tuple(emb.shape)}  <- feed these to any downstream model")
    print(f"  mlm logits     {tuple(out['logits'].shape)}")

    # ---- 2. the fine-tuned head: a loan history -> a default score --------------------
    head_dir = HERE / "finetuned-default-12m"
    if not (head_dir / "model.safetensors").exists():
        print("\\n(fine-tuned head not present locally — download the whole repo to run part 2)")
        return
    head, head_cfg = load(head_dir)
    with torch.no_grad():
        score = torch.softmax(head.classify(batch), dim=-1)[:, 1]
    task = head_cfg["finetune"]["task"]
    print(f"\\nfine-tuned head: {task['label']} "
          f"({task['label_col']} within {task['horizon_months']} months, "
          f"gated on {task['gate_col']})")
    print(f"  score per loan {[round(float(s), 5) for s in score]}")
    print("  NOTE: these are ranking scores, not calibrated probabilities of default —")
    print("  the fit set was negative-downsampled (20 negatives per positive), so the")
    print("  absolute level is inflated. Use the framework's calibration stage to map")
    print("  scores to PD, fitted on a window that does not touch your test period.")


if __name__ == "__main__":
    main()
'''


_PUBLIC_RUN_KEYS = ('run_name', 'seed', 'model', 'optimizer', 'schedule', 'runtime')


def _public_run_config(rc):
    """Provenance fit to publish: the recipe, without any storage location.

    The raw pretrain lineage embeds bucket URLs, the service-account key path and per-asset
    config filenames. None of that helps a downstream user, and none of it should travel with
    published weights, so only the reproducible hyperparameters are kept.
    """
    if not isinstance(rc, dict):
        return None
    pub = {k: rc[k] for k in _PUBLIC_RUN_KEYS if k in rc}
    if isinstance(pub.get('runtime'), dict):
        pub['runtime'] = {'bf16': pub['runtime'].get('bf16')}
    if isinstance(pub.get('schedule'), dict):
        pub['schedule'] = {k: v for k, v in pub['schedule'].items()
                           if k in ('steps', 'grad_accum', 'warmup', 'val_every')}
    return pub


def _public_finetune(ft):
    """Task definition and measured result for a fine-tuned checkpoint.

    Mirrors :func:`_public_run_config`: what the task was and how it scored is useful to a
    downstream reader; where the artefacts happened to live is not.
    """
    if not isinstance(ft, dict):
        return None
    out = {k: ft[k] for k in ('mode', 'task', 'metrics') if k in ft}
    base = ft.get('base_checkpoint')
    if base:
        out['base_checkpoint'] = Path(str(base)).name
    return out


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/mortgage_performance/publish.yaml")
    print(f"config: {cfg.config_path}\n{summarize(cfg, 'checkpoint', 'tokenizer', 'model_card', 'out')}",
          flush=True)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) load checkpoint (local or gs://) and split weights / config
    storage.ensure_auth(cfg.checkpoint, cfg.key)
    with fsspec.open(cfg.checkpoint, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    state = {k: v.contiguous() for k, v in ckpt["model"].items()}
    mc = ckpt["config"]

    # 2) weights as safetensors (no pickle)
    save_file(state, str(out / "model.safetensors"),
              metadata={"format": "pt", "framework": "credit_fm"})
    n_params = sum(v.numel() for v in state.values())

    # 3) architecture config + provenance
    config = {**mc,
              "architecture": "CreditFoundationModel (encoder-MLM, three-branch KVT)",
              "n_parameters": int(n_params),
              "framework": "credit_fm",
              "license": "Apache-2.0",
              "source_checkpoint": Path(str(cfg.checkpoint)).name,
              "run_config": _public_run_config(ckpt.get("run_config")),
              "pretrain_steps": ckpt.get("steps"),
              "finetune": _public_finetune(ckpt.get("finetune"))}
    config = {k: v for k, v in config.items() if v is not None}  # drop stage-inapplicable fields
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    # 4) tokenizer, load example, license, model card
    shutil.copyfile(cfg.tokenizer, out / "tokenizer.json")
    (out / "load_example.py").write_text(_LOAD_EXAMPLE)
    if Path("LICENSE").exists():
        shutil.copyfile("LICENSE", out / "LICENSE")
    card = Path(cfg.model_card)
    if card.exists():
        shutil.copyfile(card, out / "README.md")
    else:
        print(f"  warning: model card {card} not found — README.md not written", flush=True)

    print(f"\npackaged {n_params/1e6:.1f}M params -> {out}/", flush=True)
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)", flush=True)
    print("\nnext — publish one of:", flush=True)
    print(f"  HF Hub:  huggingface-cli upload <org>/credit-fm-mortgage-m5 {out} .", flush=True)
    print(f"  Git LFS: git lfs track '{out}/*.safetensors' && git add {out} && git commit", flush=True)


if __name__ == "__main__":
    main()
