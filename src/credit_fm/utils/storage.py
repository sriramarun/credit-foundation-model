# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Pluggable storage I/O over fsspec.

Every location is a URL, so the backend is swappable by changing the scheme — local paths,
``gs://`` (GCS), or ``s3://`` (AWS, future) — with no code change. Credentials come from the
environment: GCS via a service-account JSON (``GOOGLE_APPLICATION_CREDENTIALS``, auto-pointed at
the container key if present); S3 via the standard AWS chain (env / profile / instance role).

Requires ``fsspec``; ``gcsfs`` for ``gs://`` and ``s3fs`` for ``s3://`` (install per backend).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

GCS_DEFAULT_KEY = "/workspace/.gcloud/credit-fm-sa.json"  # service-account key on the container


def ensure_auth(url: str, key: str | None = GCS_DEFAULT_KEY) -> None:
    """If ``url`` is ``gs://`` and a key file exists, point gcsfs at it (idempotent, no-op else)."""
    if str(url).startswith("gs://") and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        if key and Path(key).exists():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key


def join(base: str, *parts: str) -> str:
    """Join a base location (local path or URL) with sub-parts, preserving any scheme.

    Unlike ``pathlib.Path``, this does not collapse ``gs://`` to ``gs:/``.
    """
    out = str(base).rstrip("/")
    for p in parts:
        out += "/" + str(p).strip("/")
    return out


def _fs(url: str, storage_options: dict[str, Any] | None):
    ensure_auth(url)
    return fsspec.core.url_to_fs(url, **(storage_options or {}))


def makedirs(url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Create a directory/prefix (no-op on object stores that lack real directories)."""
    fs, path = _fs(url, storage_options)
    fs.makedirs(path, exist_ok=True)


def write_parquet(df: pd.DataFrame, url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Write a DataFrame to ``url`` (local/gs:///s3://).

    Streams through the fsspec file handle rather than ``df.to_parquet(url)`` so it works even when
    this pyarrow build was compiled without native cloud-filesystem (GCS/S3) support.
    """
    fs, path = _fs(url, storage_options)
    parent = path.rsplit("/", 1)[0]
    if parent and parent != path:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "wb") as f:
        df.to_parquet(f, index=False)


def read_parquet(url: str, columns=None, storage_options: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read parquet — a single file or a partitioned directory — from local/gs:///s3://.

    Uses the fsspec filesystem (gcsfs/s3fs) for IO, so it works when pyarrow lacks native cloud
    support; pyarrow only parses the bytes.
    """
    if "://" not in str(url):
        return pd.read_parquet(url, columns=columns)        # local: plain path, fastest
    import pyarrow.dataset as pds
    from pyarrow.fs import FSSpecHandler, PyFileSystem
    fs, path = _fs(url, storage_options)
    dataset = pds.dataset(path, filesystem=PyFileSystem(FSSpecHandler(fs)), format="parquet")
    return dataset.to_table(columns=columns).to_pandas()


def write_text(text: str, url: str, storage_options: dict[str, Any] | None = None) -> None:
    """Write a text file to ``url`` (local/gs:///s3://), creating the parent prefix first."""
    ensure_auth(url)
    parent = url.rsplit("/", 1)[0]
    makedirs(parent, storage_options)
    with fsspec.open(url, "w", **(storage_options or {})) as f:
        f.write(text)


def sha256(url: str, storage_options: dict[str, Any] | None = None) -> str:
    """Stream a file from ``url`` and return its SHA-256 hex digest (backend-agnostic)."""
    ensure_auth(url)
    h = hashlib.sha256()
    with fsspec.open(url, "rb", **(storage_options or {})) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
