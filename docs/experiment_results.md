# Experiment results

## Confirmed leaderboard results

| Submission | Public RMSLE | Decision |
|---|---:|---|
| LightGBM, all 15 anchors, 31 leaves, child 2000, 965 trees | **1.65408** | champion |
| LightGBM plus per-anchor rank features | 1.661152 | rejected |
| Recent-anchor model plus annual stack | 1.69530 | rejected |
| Recent-anchor model plus full affine calibration | 1.696365 | rejected |

The rejected variants are useful evidence: late-fold gains did not reliably
transfer to the hidden test period. This led to a champion-first policy where
each experiment changes only one factor and must pass distribution and fold
guards before a CSV is created.

## Frozen time validation

| Validation anchor | LightGBM RMSLE | Trees |
|---|---:|---:|
| 2025-12-03 | 1.755831 | 1484 |
| 2025-12-17 | 1.743131 | 1339 |
| 2025-12-31 | 1.726136 | 417 |
| 2026-01-14 | 1.708784 | 154 |
| Mean | **1.733471** | — |

The official previous-30-days benchmark averaged 2.216177 RMSLE. A diagnostic
self-affine baseline averaged 1.933686, but calibration was not promoted because
of transfer risk.

## Reproducibility audit

| Threads | Correlation with saved champion | Mean absolute log difference | Runtime |
|---:|---:|---:|---:|
| 14 | 0.9999996184 | 0.0006755562 | 4.33 min |
| 4 | **1.0000000000** | **1.56e-17** | 10.26 min |

Four threads are retained as the deterministic default. The faster run is not
mixed into seed or capacity comparisons.

## Pre-registered diversity gate

A new model advances if standalone mean OOF improves by at least 0.002, or if a
common-OOF blend improves by at least 0.003 with mean residual correlation at
most 0.995 and no fold regression greater than 0.003.
