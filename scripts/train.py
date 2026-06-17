"""Phase 4: pretrain Credit-TFM-{S,M} on 8x H100 via torchrun.

Example:
  torchrun --nproc_per_node=8 scripts/train.py --config configs/pretrain_credit_decoder.yaml
"""
import argparse

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    raise NotImplementedError("Phase 4: pretraining loop")

if __name__ == "__main__":
    main()
