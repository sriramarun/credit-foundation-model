# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Encode a processed panel split into token-id shards (M2 data layer, encode-once).

Loads a frozen ``tokenizer.json``, reads a per-loan monthly panel (local or ``gs://``), and writes
sharded parquet where each row is one loan with aligned ragged columns
(``input_ids``/``event_index``/``field_type``/``branch``) + ``n_tokens``/``n_events``. A
``manifest.json`` records the resolved config, loan/token counts, and shard list.

Config-driven (recipe: ``configs/fannie_mae/encode.yaml``)::

    python scripts/encode_dataset.py -c configs/fannie_mae/encode.yaml            # train split
    python scripts/encode_dataset.py -c configs/fannie_mae/encode.yaml --split val
    python scripts/encode_dataset.py -c configs/fannie_mae/encode.yaml --workers 8
"""

from __future__ import annotations

import json
import time

from credit_fm.data.encode import encode_to_shards
from credit_fm.tokenizer import KVTTokenizer
from credit_fm.utils import storage
from credit_fm.utils.config import parse_cli, summarize


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/fannie_mae/encode.yaml")
    print(f"config: {cfg.config_path}\n{summarize(cfg, 'split', 'input', 'output', 'shard_size', 'workers')}",
          flush=True)

    storage.ensure_auth(cfg.input, cfg.key)
    storage.ensure_auth(cfg.output, cfg.key)
    tok = KVTTokenizer.load(cfg.tokenizer)
    print(f"loaded tokenizer ({tok.vocab_size:,} tokens) <- {cfg.tokenizer}", flush=True)
    print(f"reading {cfg.input} ...", flush=True)
    panel = storage.read_parquet(cfg.input)
    print(f"encoding {panel[tok.id_col].nunique():,} loans with workers={cfg.workers} ...", flush=True)

    t0 = time.time()
    shards, n_loans, n_tokens = encode_to_shards(
        tok, cfg.tokenizer, panel, cfg.output, shard_size=cfg.shard_size,
        workers=cfg.workers, key=cfg.key)

    manifest = {
        "tokenizer": cfg.tokenizer, "vocab_size": tok.vocab_size,
        "source": cfg.input, "n_loans": n_loans, "n_tokens": n_tokens,
        "n_shards": len(shards), "shard_size": cfg.shard_size, "shards": shards,
        "columns": ["input_ids", "event_index", "field_type", "branch", "n_tokens", "n_events"],
        "config": cfg.to_dict(),                                   # lineage
    }
    storage.write_text(json.dumps(manifest, indent=2, default=str),
                       storage.join(cfg.output, "manifest.json"))
    print(f"done: {n_loans:,} loans, {n_tokens:,} tokens, {len(shards)} shards "
          f"-> {cfg.output}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
