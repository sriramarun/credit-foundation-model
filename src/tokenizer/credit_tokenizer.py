"""Tokenizer interface for decoder-only credit model training and inference.

Credit analogue of NVIDIA's FinancialTabularTokenizer. Wraps
CreditTokenizerPipeline in the API that the CLM dataset and
train_decoder_model.py expect: encode(), decode(), vocab_size, and
special-token IDs (<bos>/<eos>/<sep>/<pad>/<unk>).
"""
from .credit_pipeline import CreditTokenizerPipeline


class CreditTabularTokenizer:
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]

    def __init__(self, pipeline: CreditTokenizerPipeline | None = None):
        self.pipeline = pipeline or CreditTokenizerPipeline()
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def build_vocab(self, df) -> None:
        raise NotImplementedError("Phase 3: fit pipeline + assign special/field token ids")

    def encode(self, df) -> list[int]:
        raise NotImplementedError("Phase 3")

    def decode(self, ids: list[int]) -> list[str]:
        raise NotImplementedError("Phase 3")

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
