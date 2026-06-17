# Deployment

Sovereign-cloud-deployable: runs entirely on customer infrastructure, no external API
dependencies (addresses GCC data-residency requirements).

- Batch scoring via `scripts/score_portfolio.py`.
- Optional FastAPI dashboard in `app/`.
- Checkpoints distributed via Git LFS under `models/`.
