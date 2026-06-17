# Credit Foundation Model

A credit-data-specific foundation model based on the
[NVIDIA Transaction Foundation Model blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model).

> **Objective:** Build a self-supervised foundation model over credit event sequences and
> demonstrate measurable lift from credit embeddings versus raw-feature baselines.

## Project at a glance

| | |
|---|---|
| Compute window | 60 days |
| Hardware | 8× NVIDIA H100 (640 GB VRAM total) |
| Recommended main model | Credit-TFM-M, 100M–250M params, 4k–8k context |
| Primary success metric | Lift from self-supervised credit embeddings vs raw-feature baselines |
| Downstream tasks | Default / delinquency, prepayment, cure, segmentation, anomaly detection |

**Definition of done:** tokenizer, trained checkpoint, embedding pipeline, downstream
benchmark, leakage audit, and model/data cards.

The full phased roadmap, experiment matrix, metrics template, risks, and deliverables
live in [`docs/project_manager_roadmap.xlsx`](docs/project_manager_roadmap.xlsx).

## Phases (60-day plan)

| Phase | Days | Focus |
|------:|------|-------|
| 0 | 1–2   | Setup and decision log |
| 1 | 3–8   | Credit event schema + leakage inventory + temporal splits |
| 2 | 9–14  | Strong tree/logistic baselines + DQ report (go/no-go gate) |
| 3 | 15–22 | Credit tokenizer v1 + corpora |
| 4 | 23–34 | Pretraining scale-up (30M → 100M–250M) |
| 5 | 35–43 | Embedding extraction + downstream eval |
| 6 | 44–50 | Tokenizer v2 + ablations + leakage audit |
| 7 | 51–56 | Prototype productization (batch embed + scoring) |
| 8 | 57–60 | Final validation + handoff (technical report, cards, demo, v2 roadmap) |

## Repository layout

```
credit-foundation-model/
├── configs/            # Tokenizer, model, and pretraining configs (YAML)
├── checkpoints/        # Trained model checkpoints (gitignored contents)
├── data/
│   ├── raw/            # Source credit datasets (gitignored)
│   └── processed/      # Observation-date feature store, corpora (gitignored)
├── docs/               # Schema, decision log, roadmap workbook
├── notebooks/          # Exploration and EDA
├── reports/            # Baselines, eval, ablations, model/data cards, handoff
├── scripts/            # CLI entry points: build corpus, train, extract, score
├── src/
│   ├── data/           # Entity schema, splits, feature store
│   ├── tokenizer/      # Credit tokenizer
│   ├── training/       # Pretraining loops / configs glue
│   └── embeddings/     # Embedding extraction + pooling
└── tests/
```

## Getting started

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Validate the 8-GPU environment (Phase 0)
python scripts/gpu_smoke_test.py

# 3. Follow the phased roadmap in docs/project_manager_roadmap.xlsx
```

## Key risks (see roadmap `Risks` tab)

- **Data leakage** from future-state fields → observation-date feature store + field-level
  leakage inventory + temporal splits.
- **Weak baselines** make foundation-model lift misleading → strong XGBoost/LightGBM baselines.
- **GPU time before data quality is proven** → enforce the Phase 2 go/no-go gate before
  large pretraining.

## Source / reference

- NVIDIA AI Blueprint: Transaction Foundation Model —
  https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
