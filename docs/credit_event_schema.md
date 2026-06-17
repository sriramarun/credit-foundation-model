# Credit Event Schema

> **Status:** DRAFT — Phase 1 deliverable (target day 8).
> Approved observation-time field list and train/val/test split design are the exit criteria.

## Entity hierarchy

```
borrower ──< loan ──< observation
```

- **borrower** — the obligor. Stable identifiers and slow-changing attributes only.
- **loan** — a credit facility belonging to a borrower (origination terms, collateral).
- **observation** — a point-in-time record for a loan at an observation date. The atomic
  unit the model sees in sequence.

## Event taxonomy

| Event | Description |
|-------|-------------|
| `observation`  | Periodic point-in-time snapshot |
| `payment`      | Scheduled / actual payment |
| `delinquency`  | Transition into a delinquency bucket (30/60/90+ DPD) |
| `default`      | Default / charge-off event |
| `cure`         | Return to performing status |
| `prepayment`   | Full or partial prepayment |
| `restructuring`| Modification of terms |

## Field inventory & leakage exclusions

Every field MUST be classified by its **observation-time availability**. Fields that encode
future state (e.g. final outcome flags, post-default recoveries) are **excluded** from
model inputs and only used to construct labels.

| Field | Entity | Type | Available at obs-time? | Notes |
|-------|--------|------|------------------------|-------|
| _TODO_ | | | | |

See the leakage inventory (`reports/` Phase 1) for the field-level audit.

## Labels

| Task | Definition | Horizon |
|------|------------|---------|
| Default 3M  | Default within 3 months of observation | 3M |
| Default 6M  | Default within 6 months of observation | 6M |
| Prepay 6M   | Prepayment within 6 months | 6M |
| Cure        | Return to performing from delinquency | TBD |

## Temporal split design

- Split by **observation date / vintage**, never randomly across borrowers.
- Train / validation / test windows must be disjoint in time to prevent leakage.
- Record exact date boundaries here once approved.
