# Extending to a New Asset Class

Adaptation is configuration, not code:
1. Convert raw data to the canonical panel (`scripts/prepare_data.py`).
2. Write `configs/<asset>/tokenizer.yaml`, `model_30m.yaml`, `training.yaml`,
   `downstream_tasks.yaml`.
3. Run the standard scripts: `train_tokenizer` → `pretrain` → `extract_embeddings` →
   `evaluate_downstream`.

The invoice-financing reference documents exactly which YAML changed vs Dutch mortgages.
