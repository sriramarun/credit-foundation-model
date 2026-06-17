"""Pre-configured tokenizer pipeline for credit event data.

Credit analogue of NVIDIA's financial_pipeline. A standard field set over the
borrower / loan / observation schema, for example:

    BAL  RATE  LTV  DTI  FICO  DPD  STATUS  PRODUCT  TERM  MOB  STATE  DT

Optional extensions:
    - amount_strategy="quantile"   data-driven balance bins
    - include_time_delta=True      inter-observation time deltas
    - include_macro=True           macro context tokens (rates, unemployment)
"""
from .pipeline import TokenizerPipeline


class CreditTokenizerPipeline(TokenizerPipeline):
    def __init__(self, amount_strategy: str = "quantile",
                 include_time_delta: bool = True, include_macro: bool = False):
        self.amount_strategy = amount_strategy
        self.include_time_delta = include_time_delta
        self.include_macro = include_macro
        super().__init__(steps=self._build_steps())

    def _build_steps(self):
        raise NotImplementedError(
            "Phase 3: assemble Numerical/Mapping/FixedVocab/Hash/TimeDelta steps "
            "from the approved credit_event_schema")
