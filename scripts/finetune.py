# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Fine-tune the pretrained FM on the forward-default task — the fair test of the FM's value.

The frozen-embedding probe (``evaluate_downstream.py``) is the *handicap* match. This adapts the
pretrained weights to the label, three ways, and reports test ROC/PR against the features bar:

* ``frozen`` — freeze the encoder, train only the classification head on cached ``[USR]`` embeddings
  (a proper task-loss linear/MLP probe; usually stronger than XGBoost-on-embeddings).
* ``lora``   — freeze the encoder, insert low-rank adapters into its linear layers + train the head
  (cheap adaptation; the standard FM-efficiency path).
* ``full``   — fine-tune the whole encoder + head at a low LR (the strongest adaptation).

Observation + label match the baseline: observe at ``--cutoff`` (performing gate, history <= cutoff,
no leakage), label = default within ``--horizon-months``. Class imbalance handled by weighted loss.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from credit_fm.data.collators import MLMCollator
from credit_fm.data.encode import encode_panel
from credit_fm.models import CreditFoundationModel
from credit_fm.tokenizer import KVTTokenizer
from credit_fm.utils import storage
from credit_fm.utils.reproducibility import set_seed

_KEYS = ("input_ids", "event_index", "field_type", "branch")


def load_checkpoint(path, key):
    storage.ensure_auth(path, key)
    with fsspec.open(path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    model = CreditFoundationModel(
        c["vocab_size"], c["n_field_types"], dim=c["dim"], n_heads=c["n_heads"],
        profile_layers=c["profile_layers"], event_layers=c["event_layers"],
        history_layers=c["history_layers"])
    model.load_state_dict(ckpt["model"])
    return model, c


def observe_panel(panel, id_col, time_col, cutoff, gate_col):
    dt = pd.to_datetime(panel[time_col], errors="coerce")
    panel = panel[dt <= pd.to_datetime(cutoff)]
    if gate_col is not None:
        last = panel.sort_values(time_col).groupby(id_col).tail(1)
        keep = set(last.loc[last[gate_col].fillna(False).astype(bool), id_col])
        panel = panel[panel[id_col].isin(keep)]
    return panel


def forward_default_loans(panel, id_col, time_col, label_col, cutoff, horizon_months):
    lo = pd.to_datetime(cutoff)
    hi = lo + pd.DateOffset(months=horizon_months)
    dt = pd.to_datetime(panel[time_col], errors="coerce")
    hit = panel[(dt > lo) & (dt <= hi) & panel[label_col].fillna(False).astype(bool)]
    return set(hit[id_col])


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update ``B @ A`` (scaled)."""

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.lora_A, std=0.02)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(model, r, alpha):
    # collect the target Linears FIRST, then wrap — mutating during traversal would re-wrap the
    # base Linear inside each new LoRALinear forever (infinite recursion).
    targets = []
    for enc in (model.profile_encoder, model.event_encoder, model.history_encoder):
        for mod in enc.modules():
            for cname, child in mod.named_children():
                if isinstance(child, nn.Linear):
                    targets.append((mod, cname, child))
    for mod, cname, child in targets:
        setattr(mod, cname, LoRALinear(child, r, alpha))


def build_samples(shard, defaulted, id_col):
    samples, ys = [], []
    for _, r in shard.iterrows():
        s = {k: torch.tensor(r[k], dtype=torch.long) for k in _KEYS}
        s["n_events"] = int(r["n_events"])
        samples.append(s)
        ys.append(int(r[id_col] in defaulted))
    return samples, np.array(ys, dtype=np.int64)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def embed_all(model, samples, collate, device, bsz, use_amp):
    model.eval()
    out = []
    for i in range(0, len(samples), bsz):
        b = _to_device(collate(samples[i:i + bsz]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out.append(model.extract_embeddings(b).float())
    return torch.cat(out)


@torch.no_grad()
def predict_full(model, samples, collate, device, bsz, use_amp):
    model.eval()
    out = []
    for i in range(0, len(samples), bsz):
        b = _to_device(collate(samples[i:i + bsz]), device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out.append(model.classify(b).float().softmax(-1)[:, 1])
    return torch.cat(out).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default="configs/fannie_mae/tokenizer.json")
    ap.add_argument("--config", default="configs/fannie_mae/tokenizer.yaml")
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--horizon-months", type=int, default=12)
    ap.add_argument("--gate-col", default="is_performing")
    ap.add_argument("--label-col", default="default_event")
    ap.add_argument("--mode", choices=["frozen", "lora", "full"], default="frozen")
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="default per mode")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", default=None)
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.bf16 and device.startswith("cuda")
    lr = args.lr or {"frozen": 1e-3, "lora": 5e-4, "full": 2e-5}[args.mode]

    cfg = yaml.safe_load(open(args.config))
    id_col, time_col = cfg["id_col"], cfg["time_col"]
    tok = KVTTokenizer.load(args.tokenizer)
    model, _ = load_checkpoint(args.checkpoint, args.key)

    storage.ensure_auth(args.panel, args.key)
    panel = storage.read_parquet(args.panel)
    defaulted = forward_default_loans(panel, id_col, time_col, args.label_col, args.cutoff,
                                      args.horizon_months)
    obs = observe_panel(panel, id_col, time_col, args.cutoff, args.gate_col)
    if args.limit:
        keep = obs[id_col].drop_duplicates().head(args.limit)
        obs = obs[obs[id_col].isin(keep)]
    shard = encode_panel(tok, obs)
    samples, y = build_samples(shard, defaulted, id_col)
    print(f"loans {len(y):,} | default rate {y.mean()*100:.2f}% | mode={args.mode} lr={lr}", flush=True)

    rng = np.random.default_rng(args.seed)
    is_test = rng.random(len(samples)) < args.test_frac
    tr_idx = np.flatnonzero(~is_test)
    te_idx = np.flatnonzero(is_test)
    tr_samples = [samples[i] for i in tr_idx]
    te_samples = [samples[i] for i in te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    n_pos = max(int(y_tr.sum()), 1)
    pos_w = (len(y_tr) - n_pos) / n_pos
    weight = torch.tensor([1.0, pos_w], device=device)
    celoss = nn.CrossEntropyLoss(weight=weight)
    collate = MLMCollator(vocab_size=tok.vocab_size, mask=False)
    model.to(device)

    if args.mode == "frozen":
        for p in model.parameters():
            p.requires_grad = False
        head = model.classification_head
        for p in head.parameters():
            p.requires_grad = True
        etr = embed_all(model, tr_samples, collate, device, args.batch_size, use_amp)
        ete = embed_all(model, te_samples, collate, device, args.batch_size, use_amp)
        ytr_t = torch.tensor(y_tr, device=device)
        opt = torch.optim.Adam(head.parameters(), lr=lr)
        head.train()
        for ep in range(args.epochs):
            order = torch.randperm(len(etr), device=device)
            for i in range(0, len(etr), args.batch_size):
                idx = order[i:i + args.batch_size]
                loss = celoss(head(etr[idx]), ytr_t[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            print(f"  epoch {ep+1}/{args.epochs}  loss {loss.item():.4f}", flush=True)
        head.eval()
        with torch.no_grad():
            probs = head(ete).float().softmax(-1)[:, 1].cpu().numpy()
    else:
        if args.mode == "lora":
            apply_lora(model, args.lora_rank, args.lora_alpha)
            model.to(device)
            for p in model.parameters():
                p.requires_grad = False
            for n, p in model.named_parameters():
                if "lora_" in n or n.startswith("classification_head"):
                    p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        print(f"  trainable params: {n_train/1e6:.2f}M", flush=True)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        ytr_t = torch.tensor(y_tr, device=device)
        model.train()
        for ep in range(args.epochs):
            order = np.random.default_rng(ep).permutation(len(tr_samples))
            last = 0.0
            for i in range(0, len(order), args.batch_size):
                idx = order[i:i + args.batch_size]
                b = _to_device(collate([tr_samples[j] for j in idx]), device)
                yb = ytr_t[torch.tensor(idx, device=device)]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = celoss(model.classify(b), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                last = loss.item()
            print(f"  epoch {ep+1}/{args.epochs}  loss {last:.4f}", flush=True)
        probs = predict_full(model, te_samples, collate, device, args.batch_size, use_amp)

    roc = roc_auc_score(y_te, probs)
    pr = average_precision_score(y_te, probs)
    print(f"\n=== Fine-tune ({args.mode}) — default within {args.horizon_months}mo of {args.cutoff} ===")
    print(f"  test: {len(y_te):,} loans, {y_te.mean()*100:.2f}% default")
    print(f"  ROC-AUC {roc:.4f} | PR-AUC {pr:.4f}   (features bar 0.746)")

    if args.report:
        rep = Path(args.report)
        rep.parent.mkdir(parents=True, exist_ok=True)
        rep.write_text(
            f"# FM fine-tune ({args.mode}) — default within {args.horizon_months}mo of {args.cutoff}\n\n"
            f"Test {len(y_te):,} loans ({y_te.mean()*100:.2f}% default), loan-disjoint.\n\n"
            f"| metric | value |\n|---|--:|\n| ROC-AUC | {roc:.4f} |\n| PR-AUC | {pr:.4f} |\n"
            f"| features bar (ROC) | 0.746 |\n")
        print(f"wrote {rep}")


if __name__ == "__main__":
    main()
