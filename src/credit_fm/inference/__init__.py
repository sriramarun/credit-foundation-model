# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Algoritmica.ai and contributors.
"""Inference + adaptation.
"""

from .extractor import EmbeddingExtractor
from .lora import attach_lora

__all__ = ['EmbeddingExtractor', 'attach_lora']
