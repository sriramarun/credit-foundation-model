# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Encode a processed panel split into token-id shards (M2 data layer, encode-once).

Loads a frozen ``tokenizer.json``, reads a per-loan monthly panel (local or ``gs://``), and writes
sharded parquet where each row is one loan with aligned ragged columns
(``input_ids``/``event_index``/``field_type``/``branch``) + ``n_tokens``/``n_events``. A
``manifest.json`` records the tokenizer, source, loan/token counts, and shard list.

Example (full corpus, parallel):
    python scripts/encode_dataset.py \
        --tokenizer configs/fannie_mae/tokenizer.json \
        --in   gs://sriram-credit-fm-data/output/processed/fannie_mae/run_2016_2017/train.parquet \
        --out  gs://sriram-credit-fm-data/output/encoded/fannie_mae/run_2016_2017/train \
        --shard-size 50000 --workers 32
"""

from __future__ import annotations

import argparse
import json
import time

from credit_fm.data.encode import encode_to_shards
from credit_fm.tokenizer import KVTTokenizer
from credit_fm.utils import storage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="configs/fannie_mae/tokenizer.json")
    ap.add_argument("--in", dest="inp", required=True, help="processed panel; local or gs:///s3://")
    ap.add_argument("--out", required=True, help="output shard directory; local or gs:///s3://")
    ap.add_argument("--shard-size", type=int, default=50_000, help="loans per shard")
    ap.add_argument("--workers", type=int, default=0,
                    help="encode shards across N processes (0/1 = in-process). Use ~#CPU cores.")
    ap.add_argument("--key", default=storage.GCS_DEFAULT_KEY)
    args = ap.parse_args()

    storage.ensure_auth(args.inp, args.key)
    storage.ensure_auth(args.out, args.key)
    tok = KVTTokenizer.load(args.tokenizer)
    print(f"loaded tokenizer ({tok.vocab_size:,} tokens) <- {args.tokenizer}", flush=True)
    print(f"reading {args.inp} ...", flush=True)
    panel = storage.read_parquet(args.inp)
    print(f"encoding {panel[tok.id_col].nunique():,} loans with workers={args.workers} ...",
          flush=True)

    t0 = time.time()
    shards, n_loans, n_tokens = encode_to_shards(
        tok, args.tokenizer, panel, args.out, shard_size=args.shard_size,
        workers=args.workers, key=args.key)

    manifest = {
        "tokenizer": args.tokenizer, "vocab_size": tok.vocab_size,
        "source": args.inp, "n_loans": n_loans, "n_tokens": n_tokens,
        "n_shards": len(shards), "shard_size": args.shard_size, "shards": shards,
        "columns": ["input_ids", "event_index", "field_type", "branch", "n_tokens", "n_events"],
    }
    storage.write_text(json.dumps(manifest, indent=2, default=str),
                       storage.join(args.out, "manifest.json"))
    print(f"done: {n_loans:,} loans, {n_tokens:,} tokens, {len(shards)} shards "
          f"-> {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
