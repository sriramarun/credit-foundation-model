# Architecture

Encoder-only (PRAGMA-style) credit foundation model with masked-language-modelling
pretraining over tabular credit panels. See the project specification for full rationale.

## Locked architectural decisions
1. **Encoder-only + MLM** (not decoder-only/causal). PRAGMA shows +130% PR-AUC on credit
   scoring; our targets are discriminative.
2. **Three-branch encoders** — Profile State (3L) + Event (4–5L) + History (4–6L). The
   dedicated Profile State Encoder gave PRAGMA +31.8% PR-AUC.
3. **Key-value-time disentangled tokenization** — preserves field identity ("LTV is 85" ≠
   "DPD is 85").
4. **Model size sized to data** — 30M default (Chinchilla-honest on ~600M synthetic tokens);
   50M only with supplementary public data; no 100M+ on synthetic-only.
5. **Apache 2.0** from day one.
6. **NeMo-compatible, not NeMo-locked** — HuggingFace is the primary stack.
