# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Dataset adapters — the code half of the dataset contract (v1.1 G1.2).

A :class:`DatasetAdapter` turns an asset's raw source into a **contract-conforming panel**
(see ``dataset_config.py``): ``id_col`` as *str*, ISO month-end time columns, label
event/gate columns present. Everything downstream of the adapter is asset-blind.

Two ways to plug in:

* ``adapter: generic`` — your panel already conforms; :class:`GenericParquetAdapter` just reads
  and checks it. **Zero code onboarding.**
* ``adapter: <name>`` — a class registered via :func:`register_adapter`, living in
  ``reference_implementations/<name>/`` (NOT in this package — the core imports no asset code;
  ``get_adapter`` imports the reference implementation lazily, by configured name).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd

from .dataset_config import DatasetConfig

REGISTRY: dict[str, type] = {}


@runtime_checkable
class DatasetAdapter(Protocol):
    """One implementation per asset; everything downstream is asset-blind."""

    config: DatasetConfig

    def load_panel(self) -> pd.DataFrame:
        """Return the contract-conforming panel (see module docstring)."""
        ...

    def sources(self) -> list[str]:
        """The raw inputs read — recorded in manifests for lineage."""
        ...


def register_adapter(name: str):
    """Class decorator: ``@register_adapter("fannie_mae")`` adds the class to the registry."""
    def wrap(cls: type) -> type:
        REGISTRY[name] = cls
        return cls
    return wrap


def get_adapter(config: DatasetConfig, **options: Any) -> DatasetAdapter:
    """Resolve ``config.adapter`` to an adapter instance.

    ``generic`` is built in. Any other name is looked up in the registry; if absent, we try
    ``import reference_implementations.<name>`` (whose import registers the class) — that keeps
    asset code out of this package while making stock scripts "just work" from the repo root.
    """
    name = config.adapter
    if name == "generic":
        return GenericParquetAdapter(config, **options)
    if name not in REGISTRY:
        try:
            import importlib
            importlib.import_module(f"reference_implementations.{name}")
        except ImportError:
            pass
    if name not in REGISTRY:
        raise KeyError(
            f"no adapter registered for '{name}' (registered: {sorted(REGISTRY) or 'none'}). "
            f"Either use adapter: generic with a conforming panel, or provide "
            f"reference_implementations/{name}/ that calls register_adapter('{name}').")
    return REGISTRY[name](config, **options)


class GenericParquetAdapter:
    """Adapter for a panel that already honors the contract — read, coerce ids, check, done."""

    def __init__(self, config: DatasetConfig, *, path: str, key: str | None = None):
        self.config = config
        self.path = path
        self.key = key

    def load_panel(self) -> pd.DataFrame:
        from credit_fm.utils import storage
        storage.ensure_auth(self.path, self.key)
        df = storage.read_parquet(self.path)
        c = self.config
        required = [c.id_col, c.time_col] + ([] if c.origination_derived else [c.origination_col])
        required += [s.event_col for s in c.labels.values()]
        required += [s.gate_col for s in c.labels.values() if s.gate_col]
        missing = sorted({col for col in required if col not in df.columns})
        if missing:
            raise ValueError(f"{self.path}: panel is missing contract columns {missing} "
                             f"(declared in {c.path or 'dataset.yaml'})")
        df[c.id_col] = df[c.id_col].astype(str)          # ids are ALWAYS strings (contract)
        return df

    def sources(self) -> list[str]:
        return [self.path]
