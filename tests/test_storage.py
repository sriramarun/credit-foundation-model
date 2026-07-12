# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""Pluggable-storage tests — prove the fsspec abstraction works for a non-local scheme.

Uses fsspec's in-memory filesystem (``memory://``) so the remote code path is exercised in CI
without any cloud credentials; the same code serves ``gs://`` and ``s3://`` by changing scheme.
"""

from __future__ import annotations

import json
import ssl

import pandas as pd
import pytest

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


# --- retry: transient cloud/network failures are retried, real errors re-raise immediately --------

def test_is_transient_classifies_network_vs_real():
    transient = [
        ssl.SSLEOFError(8, "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred"),
        Exception("HTTPSConnectionPool(host='oauth2.googleapis.com'): Max retries exceeded"),
        Exception("google.auth.exceptions.TransportError: token endpoint 503"),
        TimeoutError("connection timed out"),
    ]
    real = [FileNotFoundError("no such object"), ValueError("schema mismatch"),
            PermissionError("403 access denied")]
    assert all(storage._is_transient(e) for e in transient)
    assert not any(storage._is_transient(e) for e in real)


def test_retry_recovers_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:                       # fail twice, then succeed
            raise ssl.SSLEOFError(8, "SSL: UNEXPECTED_EOF")
        return "ok"

    assert storage.retry(flaky, tries=5, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_real_error_without_retrying():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("schema mismatch")      # not transient → must not retry

    with pytest.raises(ValueError):
        storage.retry(broken, tries=5, base_delay=0.0)
    assert calls["n"] == 1


def test_retry_gives_up_after_max_tries():
    calls = {"n": 0}

    def always_flaky():
        calls["n"] += 1
        raise TimeoutError("timed out")          # transient, but never recovers

    with pytest.raises(TimeoutError):
        storage.retry(always_flaky, tries=3, base_delay=0.0)
    assert calls["n"] == 3                        # exactly `tries` attempts, then propagate
