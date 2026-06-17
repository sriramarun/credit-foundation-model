# Repository Structure

This project mirrors the
[NVIDIA Transaction Foundation Model blueprint](https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model),
adapted to the credit domain. The design philosophy: a numbered set of notebooks at the
root **is** the end-to-end workflow, and they call into a reusable `src/` library. Configs,
data, and model artifacts sit alongside; project governance lives in `docs/` and `reports/`.

## The mental model

```
notebooks (01 → 05)   =  the recipe, run in order
        │ import
        ▼
src/      =  the ingredients (tokenizer, dataset, inference logic)
configs/  =  the dial settings
data/ + models/   =  what goes in and comes out
docs/ + reports/  =  the paperwork (schema, governance, results)
```

## Top-level layout

```
credit-foundation-model/
├── 01_dataset_baseline.ipynb            # Phase 1-2: splits + baselines
├── 02_seq_preproc_tokenization.ipynb    # Phase 3: sequences + tokenizer + corpora
├── 03_foundation_model_training.ipynb   # Phase 4: pretraining
├── 04_inference_embedding_extraction.ipynb  # Phase 5: embeddings
├── 05_downstream_credit_eval.ipynb      # Phase 5: downstream lift vs baselines
├── src/            # the importable library the notebooks call into
├── scripts/        # CLI entry points (batch / non-notebook runs)
├── configs/        # all tunable settings (YAML)
├── data/           # datasets (gitignored)
├── models/         # trained weights + checkpoints (gitignored)
├── assets/         # diagrams / figures
├── docs/           # design & governance
├── reports/        # deliverable outputs
├── tests/          # pytest tests
└── .github/, SECURITY.md, .gitattributes, requirements.txt
```

## 1. Notebooks (root) — the workflow

These mirror NVIDIA's `01`–`05` and each maps to a roadmap phase. They are **thin
orchestration** — the real logic lives in `src/`.

| Notebook | Phase | What it does |
|----------|-------|--------------|
| `01_dataset_baseline` | 1–2 | Load credit data, build temporal train/val/test splits, train XGBoost/LightGBM baselines |
| `02_seq_preproc_tokenization` | 3 | Turn data into event sequences, tokenize, write the decoder corpus |
| `03_foundation_model_training` | 4 | Pretrain Credit-TFM-S → Credit-TFM-M |
| `04_inference_embedding_extraction` | 5 | Load a checkpoint, extract embeddings |
| `05_downstream_credit_eval` | 5 | Compare raw vs embeddings-only vs raw+embeddings |

## 2. `src/` — the importable library

Flat layout (like NVIDIA's), not split into `data/training/embeddings` subfolders.

```
src/
├── clm_data.py          # build_credit_clm_dataset() — corpus → next-token samples
├── decoder_inference.py # load_model() + extract_embeddings() with pooling
│                        #   (last-token / mean / event-anchor / window)
└── tokenizer/           # the modular tokenizer package ↓
```

### `src/tokenizer/` — modular tokenizer

A tokenizer is a **pipeline of small steps**, each handling one kind of field and turning
it into token strings like `BAL_7` or `DPD_30`.

```
tokenizer/
├── base.py             # BaseTokenizer — abstract contract (build_vocab + tokenize + vocab)
├── pipeline.py         # TokenizerPipeline — runs an ordered list of steps
├── credit_pipeline.py  # CreditTokenizerPipeline — THE preset credit field set
│                       #   (BAL, RATE, LTV, DTI, FICO, DPD, STATUS, ...)
├── credit_tokenizer.py # CreditTabularTokenizer — encode()/decode()/vocab_size API
│
│   ── individual step types ──
├── numerical.py        # continuous fields → bins (balance, rate, LTV, FICO)
├── mapping.py          # low-cardinality categories (product, state, status)
├── categorical_hash.py # high-cardinality categories (servicer, MSA) → hash buckets
├── fixed_vocab.py      # bounded integers (months-on-book, term, DPD bucket)
└── timedelta.py        # time between events (payment cadence, gaps)
```

**How it composes:** `credit_pipeline` assembles a list of step objects → `pipeline` runs
them in order → `credit_tokenizer` wraps the result with special tokens
(`<bos>/<eos>/<sep>/<pad>/<unk>`) and the `encode`/`decode` API. To change what the model
sees, edit the step list in `credit_pipeline.py`.

## 3. `scripts/` — command-line entry points

```
scripts/
├── train_decoder_model.py       # pretraining launcher (torchrun; mirrors NVIDIA)
├── build_corpus.py              ┐
├── extract_credit_embeddings.py │ Phase 7 "notebooks → scripts" deliverables
├── score_credit_portfolio.py    │ (batch / productionized versions)
└── gpu_smoke_test.py            ┘ Phase 0 environment check
```

## 4. `configs/` — settings, separated from code

```
configs/
├── pretrain_credit_decoder.yaml  # NeMo AutoModel-style: LlamaConfig (GQA, RoPE),
│                                 #   batch, optimizer, checkpoint dir
├── credit_tokenizer.yaml         # which fields, bucket counts, context length
└── experiments.yaml              # the E1–E6 matrix from the roadmap
```

`pretrain_credit_decoder.yaml` is the credit twin of NVIDIA's
`pretrain_financial_decoder.yaml` — same structure, so anyone who knows the blueprint can
read it.

## 5. Artifacts & data (gitignored)

```
data/
├── raw/             # source credit files as received
├── processed/       # observation-date feature store
└── decoder_corpus/  # tokenized train/val/test corpus .txt (read by the training config)
models/
└── credit-decoder-model/   # trained weights; */checkpoints/ gitignored
```

`data/decoder_corpus/` matches the path the training config reads
(`--dataset.data_path data/decoder_corpus/train_corpus.txt`); checkpoints land under
`models/` exactly like NVIDIA.

## 6. Project governance (additions beyond the blueprint)

```
docs/      # credit_event_schema.md, decision_log.md, the roadmap .xlsx
reports/   # baseline_report, downstream_eval, ablation_memo,
           #   model_card, data_card, final_handoff/
tests/     # pytest tests
```

## 7. Repo plumbing

- `.gitattributes` — routes model weights (`*.safetensors`) through git-LFS
- `.github/workflows/ci.yml` — runs `ruff` + `pytest` on push
- `SECURITY.md` — data-handling policy (credit data is sensitive)
- `requirements.txt` — dependencies

## Mapping to the NVIDIA blueprint

| NVIDIA TFM | This project |
|------------|--------------|
| `05_xgboost_fraud_detection.ipynb` | `05_downstream_credit_eval.ipynb` |
| `src/clm_data.py` | `src/clm_data.py` (`build_credit_clm_dataset`) |
| `src/decoder_inference.py` | `src/decoder_inference.py` |
| `src/tokenizer/financial_pipeline.py` | `src/tokenizer/credit_pipeline.py` |
| `src/tokenizer/financial_tokenizer.py` | `src/tokenizer/credit_tokenizer.py` |
| `configs/pretrain_financial_decoder.yaml` | `configs/pretrain_credit_decoder.yaml` |
| `scripts/train_decoder_model.py` | `scripts/train_decoder_model.py` |

The tokenizer step modules (`base`, `pipeline`, `fixed_vocab`, `mapping`,
`categorical_hash`, `numerical`, `timedelta`) keep the same names as the blueprint.
