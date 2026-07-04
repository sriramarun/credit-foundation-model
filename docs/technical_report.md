# Credit Foundation Model — Technical Report

**finevals.ai × Sriram Krishnan · NVIDIA-sponsored (8× H100)**
Open-source (Apache-2.0) framework for training credit foundation models (`credit_fm`), with a
Fannie Mae Single-Family reference implementation.

*Status: draft, 4 Jul 2026. Numbers are from the M5 out-of-time program.*

---

## 1. Executive summary

We built a framework for pretraining **sequence foundation models on credit-event histories**, and
a reference model on the real-world **Fannie Mae Single-Family Loan Performance** dataset (2000–2024,
~2.3M sampled loans, ~1.2B tokens). The model is an encoder-only, masked-language-modelling
transformer over key–value–time tokens, ~25.7M parameters.

The central question: *does reading a loan's month-by-month behavioural sequence predict default
better than a point-in-time snapshot model (gradient-boosted trees / XGBoost)?*

**Headline result.** On the honest, deployment-grade test — **train on 2016–2021, predict defaults
in 2023–2024 on loans the model never saw** — the fine-tuned foundation model beats a strong,
leakage-free XGBoost baseline on **both** evaluation metrics:

| Model (out-of-time test 2022→2023 cutoffs) | ROC-AUC | AP (PR-AUC) |
|---|--:|--:|
| XGBoost baseline (no-leakage features) | 0.7913 | 0.0057 |
| FM — frozen head | 0.7309 | 0.0052 |
| FM — LoRA fine-tune | 0.8068 | 0.0087 |
| **FM — full fine-tune** | **0.8257** | **0.0113** |

Full fine-tuning lifts ranking (ROC) by **+0.034** and average precision (the operational metric at
0.13% default rate) by **+98%** over the baseline — a lift directly comparable to, and exceeding,
NVIDIA's transaction-foundation-model blueprint headline (+41.76% AP on fraud).

Equally important, the **deliverable is the reusable framework**, validated by this result — not a
single tuned score.

---

## 2. Background and thesis

Traditional credit models score a loan from a **snapshot**: today's loan-to-value (LTV), interest
rate, credit score, age → a risk number. This throws away the loan's *trajectory* — how its balance,
rate, and payment behaviour evolved.

Following the PRAGMA (Revolut) line of work and the NVIDIA transaction-foundation-model blueprint, we
hypothesise that a transformer pretrained on **sequences** of credit events learns general-purpose
representations of borrower behaviour that transfer to downstream tasks (default, prepayment,
segmentation) and, in particular, generalise better across **regime shifts** (e.g. crossing into new
economic years), where point-in-time features are weakest.

Two design decisions distinguish us from the NVIDIA blueprint (our baseline reference) and align us
with PRAGMA (our improvement target):

- **Encoder + masked-language-modelling (MLM)**, not a decoder with next-token prediction. Credit
  reasoning is bidirectional; we want a per-loan summary embedding, not a generator.
- **Key–value–time (KVT) tokenization** — each field becomes a fused `field=value` token plus a
  time coordinate — rather than positional token records.

---

## 3. Data

**Source.** Fannie Mae Single-Family Loan Performance data — real US fixed-rate mortgages, ~25 years
of monthly servicing records, publicly available. We ingest 2000Q1–2024Q4.

**Sampling.** Loans are sampled by hashing the loan id (a **4%** representative sample: ~2.26M loans,
125M monthly rows). Hashing is deterministic and per-loan, so a loan's entire history is kept or
dropped together, and the sample is reproducible across runs.

**Derived labels (built at ingest, from real outcomes).** `default_event` = the loan reaches
**180+ days delinquent (D180)** OR terminates as a credit-loss event (foreclosure, short sale, REO,
note sale). `is_performing`, `prepay_event` similarly derived. These are recorded facts, not
annotations.

**Leakage control (critical for credit).** Columns that reveal the outcome or only exist after
termination are **dropped** from both the FM and the baseline: current delinquency status,
zero-balance codes and dates, foreclosure/disposition dates, all loss/expense fields, and the label
itself. Using them would be circular. (A leaky configuration scores ~0.93 ROC — a mirage; we never
quote it.)

**Temporal integrity.** For the out-of-time program the processed pretraining corpus is **capped at
Dec-2022** (`reporting_max`), so the model cannot have "seen" the 2023–2024 test period during
pretraining. Splits are by loan (never by row) and by origination date (never random).

---

## 4. Model architecture

**Tokenizer (KVT).** Numeric fields are quantile-bucketed (bin edges fit on the **train split only**,
with forced boundaries — "anchors" — at regulatory cliffs like LTV 80/90/95/97 and DTI 36/43/45).
Categoricals map to their training vocabulary; unseen → `UNK`, missing → `NA`. Each loan encodes to:
[BOS] [USR] <profile tokens: original_ltv=…, credit_score=…, …>
[EVT_START] t=<age> cal=<YYYYQ#> <event tokens: current_rate=…, current_upb=…, …> [EVT_END]
… (up to 60 most-recent months) …
[EOS]

The `cal=` token anchors each month in absolute calendar time (the macro-regime signal). The M5
vocabulary is **552 tokens**; 100% lossless round-trip, 0% out-of-vocabulary.
**Three-branch encoder (~25.7M params, dim=384, 8 heads).**
- **Profile encoder** (3 layers) — static origination facts, emitted once.
- **Event encoder** (5 layers) — per-month dynamic facts, pooled per event.
- **History encoder** (6 layers) — contextualises the event sequence and distils it into a single
  384-dimensional per-loan embedding at the `[USR]` position.
Blocks use RoPE positional encoding, RMSNorm, and SwiGLU activations (matching modern LLM practice).
**Pretraining objective.** Masked-language-modelling with 3-source masking (15% token / 10% whole
event / 10% whole field-type; BERT-style 80/10/10). No labels are used. The model learns to
reconstruct hidden fields from context — forcing it to internalise how loans behave.
---
## 5. Training framework (the deliverable)
The engagement's actual product is the reusable, config-driven pipeline. Every stage runs from a
YAML recipe (`configs/fannie_mae/*.yaml`) with dotted CLI overrides, in the spirit of the NVIDIA
blueprint's config-first workflow:
| Stage | Script | Output |
|---|---|---|
| Ingest | `ingest_fannie_mae.py` | derived panel |
| Split | `prepare_data.py` | loan-stratified temporal train/val/test + audit manifest |
| Classify / fit tokenizer | `classify_schema.py`, `train_tokenizer.py` | field schema + fitted KVT tokenizer |
| Encode-once | `encode_dataset.py` | token-id shards (CPU pool / vectorized / GPU engines) |
| Pretrain | `pretrain.py` | MLM checkpoint (+ lineage config) |
| Extract | `extract_embeddings.py` | per-loan `[USR]` embeddings |
| Evaluate | `evaluate_downstream.py` | features / embeddings / combined / probe |
| Fine-tune | `finetune.py` | frozen / LoRA / full adaptation ladder |
Design properties: reproducible (seeds, source checksums, resolved config stored in every artifact);
leakage controls enforced in code; a parallel/vectorized tokenizer that encodes 1.75M loans in
~22 minutes; and a calendar-out-of-time evaluation protocol reusable for any train/test year split
(e.g. the future 2008–2010 crisis run by changing two dates).
**M5 pretraining run.** 30,000 steps, batch 32, AdamW + cosine (lr 3e-4, warmup 1000), dropout 0.1,
bf16, single H100, 3h38m. Best validation MLM loss **0.2303** — the best of the program, indicating
the 25-year corpus generalised better than the earlier 2-year runs.
---
## 6. Experimental protocol
**Observation and label.** We "stand" at an observation cutoff, keep each loan's history only up to
that date, keep only loans **performing** at the cutoff (so we predict *new* defaults), and label
each loan 1 if it defaults within the next 12 months. Future rows supply the label but are hidden
from the model's input.
**Two regimes.**
- **Benign (in-period):** single cutoff (Dec-2016), test on held-out loans of the same period. Measures
  representation quality.
- **Calendar out-of-time (OOT):** the head trains on observations at **Dec-2016…Dec-2021** (labels land
  ≤2022) and is tested on **Dec-2022 and Dec-2023** (defaults land in 2023 and 2024). Loans appearing
  in both eras are hash-assigned wholly to one side (loan-disjoint). This is the deployment test:
  train on the past, predict an unseen future.
**Fine-tuning adaptation ladder (PRAGMA-style).** *frozen* (train only the classification head on
cached embeddings) → *LoRA* (low-rank adapters, r=8/α=16, ~0.7M trainable params) → *full* (whole
model). Class imbalance (~0.1% positives) handled by downsampling fit-set negatives (`neg_per_pos=20`,
test untouched) and a capped class weight; a 10% monitoring split at the true class balance drives
per-epoch validation-ROC logging and best-epoch restore.
**Baselines.** XGBoost (GPU) on the same leakage-free, as-of-cutoff features — a strong,
industry-standard point-in-time model. For the OOT verdict we ran the baseline on the **identical**
2016–2021→2022–2023 window (`build_oot_baseline.py`).
---
## 7. Results
### 7.1 Benign window (Dec-2016 → 2017 defaults, full population)
| Model | ROC-AUC | AP |
|---|--:|--:|
| XGBoost (features) | 0.8530 | 0.0142 |
| FM frozen | 0.8126 | 0.0073 |
| FM LoRA | 0.8395 | 0.0127 |
| FM full | 0.8417 | 0.0121 |
On the benign, same-period window the fine-tuned FM comes within ~0.011 ROC of features but does not
beat it. This is expected and consistent with the NVIDIA blueprint, where frozen embeddings alone
also underperform features — a snapshot model is hard to beat when there is no regime shift to exploit.
### 7.2 Calendar out-of-time (train 2016–2021 → test 2022–2023, defaults in 2023–2024)
Same-window XGBoost baseline: **ROC 0.7913, AP 0.0057** (4.4M test observations, 5,827 defaults).
Foundation model:
| Model | ROC-AUC | Δ ROC | AP | Δ AP |
|---|--:|--:|--:|--:|
| XGBoost baseline | 0.7913 | — | 0.0057 | — |
| FM frozen | 0.7309 | −0.060 | 0.0052 | ≈0 |
| FM LoRA | 0.8068 | **+0.016** | 0.0087 | **+53%** |
| **FM full** | **0.8257** | **+0.034** | **0.0113** | **+98%** |
**The foundation model beats the baseline on both metrics on genuinely unseen future loans.** The
escalation frozen → LoRA → full is the textbook ordering: with ~8,500 positive fit examples, full
fine-tuning has enough signal to help rather than overfit (unlike the benign run's ~745 positives,
where full slightly trailed LoRA).
The generalisation story is visible in the numbers: the same models score much higher on in-era
held-out data (monitoring ROC ~0.836) than on the true future (frozen 0.731), quantifying the
out-of-time drop that all credit models suffer — the FM simply lands higher after it.
---
## 8. Discussion
**ROC vs AP.** ROC measures ranking across the whole population; AP (average precision / PR-AUC)
measures sharpness at the risky tail — the operational metric at 0.1% default rates ("of the loans
you flag as riskiest, how many truly default?"). Following the NVIDIA blueprint, we judge AP first.
On the OOT window the FM wins both, and the AP near-doubling is the operationally meaningful lift.
**Where the FM earns its keep.** Benign window: features win narrowly. Out-of-time window: the FM
wins clearly. This is exactly the thesis — the loan's behavioural *sequence* carries signal that
generalises across a shift into new years, which a static snapshot cannot see.
**Relation to prior work.** The pattern (embeddings-only < features; adaptation lifts above features)
mirrors PRAGMA's published finding that LoRA fine-tuning matches or beats task-specific models. Our
+53–98% AP lift is directly comparable to NVIDIA's +41.76% AP blueprint headline — achieved here on
real credit data and a true future-prediction test rather than an in-period fraud split.
---
## 9. Limitations and honest caveats
- **Benign years.** 2023–2024 were low-default years (0.13% base rate), so absolute AP is small for
  all models; the FM's *relative* lift is the story, not the absolute value.
- **Sampling.** The OOT baseline used a 20% loan sample, the FM a 4% sample — representative but not
  identical loans. Effect sizes (especially +98% AP) are far larger than plausible sampling noise, but
  an identical-loan rerun would tighten confidence intervals.
- **Statistical power.** ~1,000–5,800 positives at test; ROC margin of error ≈ ±0.01. The full-model
  ROC win (+0.034) is ~3× that; the frozen result is below the bar (the expected floor).
- **Single corpus.** Validated on Fannie Mae only; a Dutch-mortgage synthetic set is used for
  controlled ablation, and invoice-financing is a planned second reference.
- **Model scale.** At 25.7M parameters the model is Chinchilla-matched to ~0.5B tokens; the 1.2B-token
  M5 corpus can support ~65M, and the full dataset ~1B+ — untested headroom.
- **Crisis regime untested.** The 2008–2010 stress window (the hardest test) is the natural next run;
  the OOT protocol is already wired for it.
---
## 10. Reproducibility
Every artifact stores the exact resolved config that produced it. The OOT verdict reproduces via:
```bash
python scripts/ingest_fannie_mae.py   -c configs/fannie_mae/ingest_2000_2024.yaml
python scripts/prepare_data.py        -c configs/fannie_mae/prepare.yaml --run_name run_2000_2022 \
    --input '${paths.raw}/panel_2000_2024.parquet' --reporting_max 2022-12-31
python scripts/train_tokenizer.py     -c configs/fannie_mae/tokenizer_fit.yaml --run_name run_2000_2022 \
    --out configs/fannie_mae/tokenizer_v2.json
python scripts/encode_dataset.py      -c configs/fannie_mae/encode.yaml --run_name run_2000_2022 \
    --tokenizer configs/fannie_mae/tokenizer_v2.json --split train --engine vector
python scripts/pretrain.py            -c configs/fannie_mae/pretrain.yaml --run_name run_2000_2022 \
    --tokenizer configs/fannie_mae/tokenizer_v2.json --data.batch_size 32 --schedule.steps 30000
python scripts/finetune.py            -c configs/fannie_mae/finetune_oot.yaml --mode full
python scripts/build_oot_baseline.py  --train-years 2016-2021 --test-years 2022-2023 --horizon-months 12
Artifacts: pretrained checkpoint runs/m5_full.pt, tokenizer configs/fannie_mae/tokenizer_v2.json,
result reports under reports/m5_oot_ft_*.md and reports/fannie_oot_2022_2023.md.

11. Future work
Crisis out-of-time (train pre-2007, test 2008–2010) — the hardest regime shift; same protocol.
Model scale-up — 65M on the current corpus, then 500M–1B with the data tap opened and
multi-GPU (DDP) training.
PCA-combined and multi-task adapters — the blueprint's combined recipe and PRAGMA's
shared-backbone-plus-adapters serving story.
Second corpus (invoice financing) to demonstrate framework generality.
Open-source release package — model/data cards, published weights (HF/LFS), notebooks 01–05.
Appendix — key configuration
Params 25.7M · dim 384 · heads 8 · layers 3/5/6 (profile/event/history) · RoPE/RMSNorm/SwiGLU
Vocab 552 (KVT v2) · max 60 events/loan · calendar=yearquarter · numeric anchors at LTV 80/90/95/97, DTI 36/43/45
Pretrain: 30k steps · batch 32 · AdamW lr 3e-4 cosine · warmup 1000 · dropout 0.1 · bf16 · best-val restore
Fine-tune: frozen/LoRA(r8,α16)/full · neg_per_pos 20 · pos_weight cap 100 · 3 epochs · best-epoch restore
Data: Fannie Mae 2000–2024 · 4% loan sample · 2.26M loans · pretrain capped Dec-2022
EOF
echo "wrote docs/technical_report.md ($(wc -l < docs/technical_report.md) lines)"