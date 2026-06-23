# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Pluggable-storage tests — prove the fsspec abstraction works for a non-local scheme.

Uses fsspec's in-memory filesystem (``memory://``) so the remote code path is exercised in CI
without any cloud credentials; the same code serves ``gs://`` and ``s3://`` by changing scheme.
"""

from __future__ import annotations

import json

import pandas as pd

from credit_fm.utils import storage


def test_join_preserves_url_scheme():
    assert storage.join("gs://bucket/prefix/", "a", "b.parquet") == "gs://bucket/prefix/a/b.parquet"
    assert storage.join("data/processed", "train.parquet") == "data/processed/train.parquet"
    assert storage.join("s3://b/p", "/x/") == "s3://b/p/x"


def test_parquet_roundtrip_non_local():
    df = pd.DataFrame({"loan_id": [1, 2, 3], "y": [0, 1, 0]})
    url = "memory://proc/train.parquet"
    storage.write_parquet(df, url)
    pd.testing.assert_frame_equal(pd.read_parquet(url), df)


def test_text_and_sha256_non_local():
    url = "memory://proc/splits.meta.json"
    payload = json.dumps({"seed": 42, "n": 3})
    storage.write_text(payload, url)
    import fsspec
    with fsspec.open(url, "r") as f:
        assert json.loads(f.read())["seed"] == 42
    assert len(storage.sha256(url)) == 64        # hex digest, backend-agnostic