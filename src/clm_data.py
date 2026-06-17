"""Causal-LM dataset builder for credit event sequences.

Credit analogue of NVIDIA's clm_data.py. Reads a decoder corpus (one
tokenized credit history per line), produces fixed-length next-token
training samples, and is referenced by configs/pretrain_credit_decoder.yaml
via `dataset._target_: src/clm_data.py:build_credit_clm_dataset`.
"""
from __future__ import annotations


def build_credit_clm_dataset(data_path: str, hash_size: int = 2000, seq_length: int = 4096):
    """Build a causal-LM dataset from a credit decoder corpus file."""
    raise NotImplementedError("Phase 3/4: load corpus, tokenize, window to seq_length")
