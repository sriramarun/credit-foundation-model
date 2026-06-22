# Credit Foundation Model Framework

An **open-source framework (Apache 2.0) for training credit foundation models** on tabular
credit panel data — plus reference implementations across asset classes. Built so banks and
financial institutions can train their own credit foundation models on their own data without
building the underlying infrastructure from scratch.

The **primary corpus is the Fannie Mae Single-Family Loan Performance dataset** — ~25 years of
real-world US fixed-rate mortgages. A synthetic Dutch RMBS panel serves as a controlled
validation/ablation set, and invoice financing is a planned third asset class.

Co-founder engagement between **finevals.ai** and **Sriram Krishnan**, sponsored by
**NVIDIA** (8× H100), 3-month delivery.

## Approach

Encoder-only (BERT-style) architecture with masked-language-modelling pretraining over
credit-event sequences, synthesizing two published references:

- **PRAGMA** (Revolut + NVIDIA, 2026) — primary architectural reference (encoder-only,
  three-branch, +130% PR-AUC on credit scoring).
- **NVIDIA Transaction Foundation Model** — reference for the training stack.

The framework is schema-agnostic: adapt to a new asset class by writing YAML (tokenizer,
model, training, downstream tasks), then running the standard scripts.

## What's in here

| Component | Location | Description |
|-----------|----------|-------------|
| Framework (`credit_fm`) | `src/credit_fm/` | Tokenizer, three-branch model, data, training, inference, evaluation |
| Fannie Mae reference (**primary**) | `configs/fannie_mae/`, `reference_implementations/fannie_mae/` | Real-world US single-family fixed-rate mortgages (~25y), 30M checkpoint |
| Dutch mortgages reference (validation) | `configs/dutch_mortgages/`, `reference_implementations/dutch_mortgages/` | ESMA Annex 2 synthetic RMBS; carries the hidden `_segment` ceiling proof |
| Invoice financing reference | `configs/invoice_financing/`, `reference_implementations/invoice_financing/` | Third asset class (data TBD) |
| Dashboard | `app/` | FastAPI demo over the four pipeline stages |

## Differentiation

1. **Open source** — weights, code, tokenizers, references all Apache 2.0 (PRAGMA is proprietary).
2. **Real-world primary corpus** — pretrained on Fannie Mae's 25-year single-family performance
   data; the Dutch validation set is ESMA Annex 2-aligned byte-for-byte.
3. **Sovereign-cloud-deployable** — runs entirely on customer infrastructure, no external APIs.

## Architecture (three-branch encoder)

```
static fields ─▶ Profile State Encoder (3L) ─┐
                                              ├─▶ History Encoder (4–6L) ─▶ [USR] embedding
per-cutoff events ─▶ Event Encoder (4–5L) ────┘
```

Pretraining: MLM with three masking sources (15% tokens / 10% events / 10% semantic types).
Downstream: embedding probe (frozen) or LoRA fine-tuning. Default size **30M**
(Chinchilla-honest on ~600M synthetic tokens).

## Quickstart

On an H100 / NGC PyTorch container, see [`docs/container_setup.md`](docs/container_setup.md)
for a restart-proof bring-up (or run `bash scripts/setup_container.sh`). Otherwise:

```bash
pip install -e .            # installs the credit_fm package
# Fannie Mae (primary), dev sample end to end:
python scripts/ingest_fannie_mae.py --gcs gs://<bucket>/<prefix> --quarters 2018Q1 2018Q2 --out data/raw/fannie_mae
python scripts/prepare_data.py --input data/raw/fannie_mae/panel.parquet --origination-col origination_date
python scripts/train_baseline.py --config configs/fannie_mae/baseline.yaml --report reports/fannie_baseline_report.md
# Dutch mortgages (validation) reference, end to end:
bash reference_implementations/dutch_mortgages/train.sh
bash reference_implementations/dutch_mortgages/evaluate.sh
```

Walkthrough notebooks: `notebooks/01_data_and_baseline` → `05_downstream_evaluation`.

## Repository layout

See [`docs/architecture.md`](docs/architecture.md) for full rationale.

```
src/credit_fm/   tokenizer/ models/ data/ training/ inference/ evaluation/ utils/
configs/         fannie_mae/ (primary) · dutch_mortgages/ · invoice_financing/   (YAML per asset class)
scripts/         ingest_fannie_mae · prepare_data · train_baseline · train_tokenizer · pretrain · …
notebooks/       01–05 pipeline walkthroughs
reference_implementations/   per-asset README, cards, train.sh, evaluate.sh
models/          pretrained checkpoints (Git LFS)
app/             FastAPI dashboard
docs/            architecture · tokenization · training · evaluation · extending · deployment
tests/           tokenizer · models · data · training · evaluation · e2e
```

## Existing assets used

- Fannie Mae [Single-Family Loan Performance Dataset](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data) (primary corpus)
- Synthetic Dutch RMBS dataset: [`Algoritmica/green-lion-2024-2025`](https://huggingface.co/datasets/Algoritmica/green-lion-2024-2025)
- [deeploans synthetic-data-designer](https://github.com/Algoritmica-ai/deeploans/tree/main/synthetic-data-designer)
- [NVIDIA TFM blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model)

## References

- PRAGMA — Ostroukhov et al., 2026 (arXiv:2604.08649)
- BERT — Devlin et al., 2019 · LoRA — Hu et al., 2021 · Chinchilla — Hoffmann et al., 2022
- ESMA Annex 2 — Commission Delegated Regulation (EU) 2020/1224

## License

Apache 2.0 — see [LICENSE](LICENSE).