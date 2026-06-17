"""Embedding extraction from a trained credit decoder model.

Credit analogue of NVIDIA's decoder_inference.py. Loads a checkpoint and
extracts sequence embeddings at prediction points using configurable pooling.
Pooling strategies: last-token, mean, event-anchor, window.
"""
from __future__ import annotations


def load_model(checkpoint_dir: str):
    raise NotImplementedError("Phase 5: load consolidated safetensors checkpoint")


def extract_embeddings(model, sequences, pooling: str = "last-token"):
    """Return one embedding vector per sequence."""
    assert pooling in {"last-token", "mean", "event-anchor", "window"}
    raise NotImplementedError("Phase 5: forward pass + pooling")
