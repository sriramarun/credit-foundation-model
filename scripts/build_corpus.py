"""Phase 3: build train/val/test decoder corpus files from tokenized sequences.

Usage: python scripts/build_corpus.py --config configs/credit_tokenizer.yaml
"""
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="data/processed/corpus")
    args = ap.parse_args()
    raise NotImplementedError("Phase 3: tokenize splits and write corpus files")

if __name__ == "__main__":
    main()
