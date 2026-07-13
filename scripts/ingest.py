# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Ingest a dataset via its adapter — the asset-blind driver (v1.1 G1.4).

Reads the recipe's ``dataset:`` pointer, resolves the asset's
:class:`~credit_fm.data.adapter.DatasetAdapter` (e.g. ``reference_implementations/fannie_mae``),
loads the contract-conforming panel, and writes ``<out>/<combined_name>``. All asset-specific
logic (column derivations, source layouts) lives in the adapter, not here.

Config-driven (recipe: ``configs/fannie_mae/ingest.yaml``)::

    python scripts/ingest.py -c configs/fannie_mae/ingest_2000_2024.yaml
    python scripts/ingest.py -c configs/fannie_mae/ingest_2000_2024.yaml \
        --sample_pct 10 --combined_name panel_2000_2024_10pct.parquet
"""

from __future__ import annotations

import os
from pathlib import Path

from credit_fm.data.adapter import get_adapter
from credit_fm.data.dataset_config import load_dataset_config
from credit_fm.utils import storage
from credit_fm.utils.config import parse_cli, summarize


def _maybe_auth(key: str | None) -> None:
    """Point gcsfs at the service-account key if one is available and not already set."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    if key and Path(key).exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key
        print(f"Using GCS key: {key}")


def main() -> None:
    cfg = parse_cli(__doc__, default_config="configs/fannie_mae/ingest.yaml")
    print(f"config: {cfg.config_path}\n"
          f"{summarize(cfg, 'dataset', 'sources', 'out', 'sample_pct', 'workers')}", flush=True)

    ds = load_dataset_config(cfg.dataset)
    if ds.adapter == "generic":
        raise SystemExit(f"dataset '{ds.name}' uses adapter: generic — its panel already "
                         "conforms; point prepare_data at it directly (nothing to ingest).")

    _maybe_auth(cfg.get_path("key"))
    out = cfg.out.rstrip("/")                           # local path or gs:///s3:// URL
    storage.ensure_auth(out, cfg.get_path("key"))

    adapter = get_adapter(ds, stage=cfg)                # asset logic lives behind the contract
    panel = adapter.load_panel()

    panel_path = storage.join(out, cfg.combined_name)
    storage.write_parquet(panel, panel_path)            # pluggable: local / gs:// / s3://
    print(f"\nWrote {panel_path}: {len(panel):,} rows, "
          f"{panel[ds.id_col].nunique():,} loans, "
          f"reporting {panel[ds.time_col].min()} -> {panel[ds.time_col].max()}, "
          f"origination {panel[ds.origination_col].min()} -> {panel[ds.origination_col].max()}")
    print(f"sources: {len(adapter.sources())} (adapter: {ds.adapter})")
    print("Next: python scripts/prepare_data.py -c configs/fannie_mae/prepare.yaml")


if __name__ == "__main__":
    main()
