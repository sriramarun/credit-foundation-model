"""Phase 5: extract embeddings at prediction points and write parquet.

Pooling strategies: last-token, mean, event-anchor, window.
Usage: python scripts/extract_credit_embeddings.py --ckpt checkpoints/<run> --pooling mean
"""
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pooling", default="mean",
                    choices=["last-token", "mean", "event-anchor", "window"])
    ap.add_argument("--out", default="data/processed/embeddings.parquet")
    args = ap.parse_args()
    raise NotImplementedError("Phase 5: embedding extraction")

if __name__ == "__main__":
    main()
