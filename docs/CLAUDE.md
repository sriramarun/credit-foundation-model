# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo. Read this first when reopening.

## What this is

An **open-source (Apache 2.0) framework for training credit foundation models** (`credit_fm`
package) plus two reference implementations (Dutch mortgages, invoice financing). Co-founder
engagement: **finevals.ai × Sriram Krishnan**, NVIDIA-sponsored (8× H100), ~12-week delivery.

Approach: **encoder-only, masked-language-modelling** over tabular credit-event sequences
(PRAGMA-style), three-branch architecture, key-value-time tokenization. The thesis: a sequence
foundation model beats point-in-time tabular baselines (XGBoost) on credit tasks.

## Architecture (locked — see `docs/decision_log.md`)

- **Encoder-only + MLM** (not decoder/causal) — DL-001.
- **Three-branch encoders**: Profile State (static fields) + Event (per-cutoff dynamic fields)
  + History (contextualizes the sequence) → `[USR]` per-loan embedding — DL-002.
- **Key-value-time tokenization**: each field → key token + value token(s) + temporal coord — DL-003.
- **30M params default** (Chinchilla-honest on ~600M synthetic tokens) — DL-004.
- **HuggingFace primary**, NeMo optional — DL-006.
- **Open question DL-009**: W&B hosted vs offline/self-hosted — resolve before pretraining
  (sovereign-cloud / data-residency requirement).

## Repo layout
src/credit_fm/ tokenizer/ (KVT) · models/ (3-branch) · data/ · training/ · inference/ · evaluation/ · utils/
scripts/ prepare_data, classify_schema, train_baseline, train_tokenizer, pretrain,
extract_embeddings, evaluate_downstream, score_portfolio, setup_container.sh
configs/ dutch_mortgages/ · invoice_financing/ (YAML per asset class)
notebooks/ 00_smoke_test_splits, 01–05 walkthroughs
reference_implementations/ per-asset README, cards, train.sh, evaluate.sh
models/ checkpoints (Git LFS) reports/ baseline_report.md, ...
docs/ architecture, tokenization, training, evaluation, decision_log, model_cards/
app/ FastAPI dashboard tests/ test_data.py (real), others stubs

Most of `src/credit_fm/` is still **scaffold** (`raise NotImplementedError`). What's real:
`data/splits.py`, `data/schema.py` (`classify_fields`, `find_redundant`), `utils/`.
## Current status (Week 1 done)
**Done & on `main`:** repo + scaffold + CI · H100 container setup · loan-stratified **temporal**
split (`prepare_data.py`) · reproducible 71-field classification → `configs/dutch_mortgages/tokenizer.yaml`
(42 features) · XGBoost baseline (`train_baseline.py`) · decision log DL-001…010.
**Baseline / Gate G1** (the bar the FM must beat): on the honest task (no-leakage features +
performing-at-observation gate, predict *new* defaults) = **ROC-AUC 0.73 / PR-AUC 0.046**.
A leaky/ungated config scores 0.93 — do **not** quote that as the baseline.
**Architectural proof:** a hidden `_segment` latent (in `loan_book.parquet`, NOT the panel)
drives a **16–32× default spread** invisible to ESMA-feature models → the XGBoost ceiling the
FM is meant to break. `train_baseline.py --book data/raw/loan_book.parquet` shows it.
**Next:** tokenizer build (`tokenizer/vocabulary.py` → `numeric_bucketer` → `categorical`/
`temporal` → `KVTTokenizer`, vocab on `train` only) → Milestone **M1**.
## Data (none committed — all gitignored)
- **Panel:** `data/raw/all_cutoffs.parquet` (canonical; 500k loans × 24 monthly cutoffs, 71
  ESMA Annex 2 cols). Source: HF `Algoritmica/green-lion-2024-2025` / deeploans generator.
  (Older `Overall_2024_2025_all_months.parquet` is a prior extract.)
- **Splits:** `python scripts/prepare_data.py --input data/raw/all_cutoffs.parquet` →
  `data/processed/{train,val,test}.parquet` + `splits.csv` + `splits.meta.json`. **Always run
  this first**; the baseline/notebook read its output.
- **Latents (eval-only):** `data/raw/loan_book.parquet` has `_segment`, `_latent_fragility`,
  `_cohort_quality`. **NEVER use these as model features** — they're ground-truth only, not in
  production. Use only for the ceiling validation. (The matching file is the generator run whose
  `_segment` predicts this panel's defaults — verify with a segment-conditional default rate.)
- **No `origination_date` column** — the temporal split derives origination =
  `reporting_date − seasoning_months` (DL-007).
## Conventions
- Python 3.10+. **`ruff`** clean + **`pytest`** green before every commit. Type hints on public
  APIs; Google-style docstrings. Every file: SPDX header + `Copyright (c) 2026 finevals.ai`.
- **Leakage rules** (critical for credit): split by `loan_id` (never row); temporal by
  origination; vocab/bins fit on `train` only (DL-008); the 8 contemporaneous-state columns
  (`arrears_bucket`, `performing_status`, `default_crr_flag`, `foreclosure_flag`, `days_past_due`,
  `arrears_amount`, `forbearance_flag`, `restructuring_flag`) are leakage for default prediction —
  the honest baseline drops them and gates to performing-at-observation.
- **Config is generated, not hand-edited**: `tokenizer.yaml` comes from `scripts/classify_schema.py`
  (its header records the regenerate command). Redundancy pruning lives in code (`find_redundant`),
  not manual edits.
## Dev workflow (IMPORTANT)
- **All git happens on the H100 container**, not locally. The user drives edits/commits there.
- **GitHub is the hub**: branch → PR → merge. Don't commit to `main` directly.
- Container bring-up: `bash scripts/setup_container.sh` (restart-proof venv under `/workspace`;
  see `docs/container_setup.md`). Secrets in `/workspace/secrets.env` (never committed).
- `git` has no pager on the box if `core.pager=cat` is set; otherwise press `q` to exit pagers.
## Internal trackers (NOT in this repo)
Project plan, backlog, dated progress, deliverables manifest, and weekly status live **outside
the repo** in `_credit-fm-internal/` (sibling folder, never pushed). Keep them in sync when
status changes. This `CLAUDE.md` is the public-repo handoff; the internal folder is the planning
source of truth.
## Gotchas learned
- `.gitignore` must anchor data rules (`data/*`, not bare `data/`) — a bare `data/` once silently
  dropped the entire `src/credit_fm/data/` module from commits.
- The synthetic panel is rule-based, so baselines run high; report the honest (gated, no-leakage)
  number and the segment-ceiling context, not the inflated one.
- NGC PyTorch image ships without `ensurepip`; the venv needs `--system-site-packages` (to keep
  the image's CUDA torch) and `pythonX.Y-venv` installed — both handled by `setup_container.sh`.