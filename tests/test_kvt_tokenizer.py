# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 finevals.ai and contributors.
"""KVTTokenizer composition tests — branch routing, sequence structure, roundtrip, save/load."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_fm.tokenizer import KVTTokenizer

CONFIG = {
    "id_col": "loan_id",
    "time_col": "reporting_date",
    "time_field": "loan_age",
    "profile": {"numeric": ["original_ltv"], "categorical": ["channel"]},
    "event": {"numeric": ["current_interest_rate", "current_upb"], "categorical": []},
    "n_bins": 8,
    "max_categories": 64,
    "max_events": 60,
}


def _panel(n_loans=4, n_months=5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for lid in range(n_loans):
        ltv = int(rng.integers(40, 97))                 # static per loan
        chan = rng.choice(["R", "C", "B"])
        rate = float(rng.uniform(3, 8))
        for m in range(n_months):
            rows.append({
                "loan_id": f"L{lid}", "reporting_date": f"2020-{m+1:02d}-28",
                "loan_age": 12 + m, "original_ltv": ltv, "channel": chan,
                "current_interest_rate": rate, "current_upb": 200_000 - m * 1_000,
            })
    return pd.DataFrame(rows)


def test_fit_builds_vocab_with_field_and_special_tokens():
    tok = KVTTokenizer(CONFIG).fit(_panel())
    assert tok.vocab_size > len(tok.vocabulary.token_to_id) - 1  # has tokens
    toks = set(tok.vocabulary.token_to_id)
    assert any(t.startswith("original_ltv=") for t in toks)
    assert any(t.startswith("channel=") for t in toks)
    assert any(t.startswith("current_interest_rate=") for t in toks)
    assert any(t.startswith("t=") for t in toks)


def test_encode_sequence_structure():
    panel = _panel(n_months=5)
    tok = KVTTokenizer(CONFIG).fit(panel)
    loan = panel[panel.loan_id == "L0"]
    seq = tok.decode(tok.encode(loan))
    assert seq[0] == "[BOS]" and seq[1] == "[USR]" and seq[-1] == "[EOS]"
    assert seq.count("[EVT_START]") == 5 == seq.count("[EVT_END]")
    assert any(t.startswith("original_ltv=") for t in seq[2:seq.index("[EVT_START]")])
    assert sum(t.startswith("t=") for t in seq) == 5


def test_roundtrip_is_lossless():
    panel = _panel()
    tok = KVTTokenizer(CONFIG).fit(panel)
    loan = panel[panel.loan_id == "L1"]
    ids = tok.encode(loan)
    assert tok.decode(ids) == tok.tokens(loan)


def test_save_load_reproduces_encoding(tmp_path):
    panel = _panel()
    tok = KVTTokenizer(CONFIG).fit(panel)
    loan = panel[panel.loan_id == "L2"]
    before = tok.encode(loan)
    p = tmp_path / "kvt.json"
    tok.save(p)
    reloaded = KVTTokenizer.load(p)
    assert reloaded.vocab_size == tok.vocab_size
    assert reloaded.encode(loan) == before


def test_unseen_value_maps_without_growing_vocab():
    panel = _panel()
    tok = KVTTokenizer(CONFIG).fit(panel)
    before = tok.vocab_size
    test_loan = pd.DataFrame([{
        "loan_id": "Lx", "reporting_date": "2021-01-28", "loan_age": 99,
        "original_ltv": 200, "channel": "ZZZ", "current_interest_rate": 999.0,
        "current_upb": 0,
    }])
    seq = tok.decode(tok.encode(test_loan))
    assert tok.vocab_size == before
    assert "channel=UNK" in seq
