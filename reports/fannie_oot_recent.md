# Fannie Mae — Out-of-Time Gate-G1 Baseline

Calendar split — **train 2000–2021**, **test 2023-2025** (val = 10% of train loans). 20% loan sample; 12-month default horizon; 57 no-leakage features. Loan split: **disjoint**; embargo dropped train years [2022].

Each loan observed every Dec it is performing; label = default (D180 / Zero-Balance credit event) within the horizon. True out-of-time: the test years are never trained on.

## Population

| Split | observations | defaults | default rate |
|---|--:|--:|--:|
| train | 34,016,360 | 117,444 | 0.35% |
| val | 3,229,149 | 11,010 | 0.34% |
| test | 5,633,579 | 5,175 | 0.09% |

## Result (out-of-time test)

| Metric | value |
|---|--:|
| ROC-AUC | 0.7844 |
| PR-AUC | 0.0030 |
| test default rate | 0.09% |

## Test default rate by year

| obs_year | observations | default rate |
|---|--:|--:|
| 2023 | 1,819,385 | 0.13% |
| 2024 | 1,886,307 | 0.14% |
| 2025 | 1,927,887 | 0.00% |

## Notes
- Real-world Fannie Mae loan performance — default rates and lift reflect an actual portfolio (no synthetic inflation). Numbers depend on the ingested reporting span and observation date.
- Out-of-time by calendar year is the honest generalization test (train on the past, score the future). The foundation model must beat this ROC/PR-AUC.
- **Guards:** loan-disjoint = `disjoint` (no loan appears in both train and test); embargo = train label windows end before the test period (no macro bleed).