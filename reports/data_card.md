# Data Card — Credit Foundation Model

> **Status:** TEMPLATE — Phase 8 deliverable (target day 59).

## Datasets
| Dataset | Source | Coverage (vintages) | Rows | Owner |
|---------|--------|---------------------|------|-------|
| _TODO_  | | | | |

## Entities & granularity
- borrower → loan → observation (point-in-time). See
  [`../docs/credit_event_schema.md`](../docs/credit_event_schema.md).

## Observation-time / leakage policy
- Inputs restricted to fields available at the observation date.
- Future-state fields used only to derive labels; never as features.
- Field-level leakage inventory: _link Phase 1 audit_.

## Splits
- Temporal train / validation / test by observation date. Exact boundaries: _TODO_.

## Known quality issues
- _Missingness, drift by vintage, class imbalance, etc._

## Licensing / access
- _Provenance, usage restrictions, PII handling._
