# Deployment

Sovereign-cloud-deployable: runs entirely on customer infrastructure, no external API
dependencies (addresses data-residency requirements).

- **Packaged checkpoints** — `scripts/publish_model.py` bundles a trained checkpoint into a
  distributable directory: `model.safetensors` + `config.json` + the frozen `tokenizer.json` +
  model card + a load example. Weights-hub publication is optional and off by default.
- **Batch scoring** — `scripts/score_portfolio.py` (deliverable #6): score a portfolio parquet
  with a fine-tuned model at an observation date. Leakage-safe (history truncated to the cutoff,
  performing-gate), writes `scores.parquet` + a manifest; `scripts/validate_scores.py` audits the
  output. The fine-tuned model comes from `finetune.py --save` (backbone + head + reload
  metadata); the shared inference path lives in `credit_fm.inference.scoring`.
- **Storage is pluggable** — every pipeline path may be local, `gs://`, or `s3://`
  (`credit_fm.utils.storage`); swapping cloud = swapping the URL scheme.
- **Environment** — a single restart-proof container bring-up (`scripts/setup_container.sh`,
  see `container_setup.md`); no services required beyond Python + a GPU for training
  (inference runs on CPU if needed).
