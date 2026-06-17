# Training

MLM pretraining with three masking sources: 15% individual tokens + 10% whole events +
10% semantic types. AdamW, lr 3e-4, 500-step warmup, cosine decay, bf16, seq length 512.

Backends: HuggingFace `Trainer` (default) or NeMo AutoModel (`--backend nemo`). Multi-GPU
on 8× H100. Every run tracked in W&B under `credit-foundation-model` with model size,
dataset, hyperparameters, git commit, and compute hours.
