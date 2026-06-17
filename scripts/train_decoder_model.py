"""Pretrain the credit decoder foundation model (Phase 4).

Credit analogue of NVIDIA's train_decoder_model.py. Driven by a NeMo
AutoModel-style YAML config.

Multi-GPU (recommended):
    torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
        -c configs/pretrain_credit_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt \
        --validation_dataset.data_path data/decoder_corpus/val_corpus.txt

Single GPU (testing):
    python scripts/train_decoder_model.py \
        -c configs/pretrain_credit_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt
"""
import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--dataset.data_path", dest="train_path")
    ap.add_argument("--validation_dataset.data_path", dest="val_path")
    args = ap.parse_args()
    raise NotImplementedError("Phase 4: build model+data from config and run CLM training")


if __name__ == "__main__":
    main()
