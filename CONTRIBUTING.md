# Contributing

Thanks for your interest in the Credit Foundation Model Framework.

## Development setup
```bash
pip install -e ".[dev,app]"
pytest -q
ruff check .
```

## Standards
- Python 3.10+. `ruff` for lint/format. Type hints on all public APIs. Google-style docstrings.
- Every `src/credit_fm/` module has a matching test in `tests/`.
- Every script accepts `--seed` and logs the seed + git commit hash.
- Conventional, descriptive commits. Open a PR against `main`; CI must pass.

## Licensing
By contributing you agree your contributions are licensed under Apache 2.0.
