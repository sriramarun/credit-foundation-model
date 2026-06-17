"""Phase 7: repeatable monthly batch embedding + scoring pipeline.

Usage: python scripts/score_credit_portfolio.py --asof YYYY-MM-DD
"""
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="Observation/as-of date")
    ap.add_argument("--out", default="reports/portfolio_scores.parquet")
    args = ap.parse_args()
    raise NotImplementedError("Phase 7: batch embed + score portfolio")

if __name__ == "__main__":
    main()
