# Architecture and design decisions

## Data flow

```text
immutable train.parquet + sample IDs
                 │
                 ▼
 point-in-time anchor generation
 15 labeled anchors + 1 test anchor
                 │
                 ▼
 Zstandard Parquet cache (5 × 50k users per anchor)
                 │
                 ├───────────────┐
                 ▼               ▼
 compact 168 features      keyed metadata + target
                 │               │
                 └───────┬───────┘
                         ▼
           frozen rolling time validation
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      LightGBM log model      XGBoost CUDA diversity
              │                     │
              └──────────┬──────────┘
                         ▼
             common keyed OOF blend gate
                         │
                         ▼
          validated CSV + JSON/SHA-256 manifest
```

## Point-in-time correctness

For anchor `A`, features may use `event_date <= A`. The target includes exactly
`A + 1 ... A + 30`. For validation anchor `V`, a training anchor is accepted
only when `A + 30 <= V`. This stronger rule prevents the training target from
overlapping the activity history available to the validation snapshot.

The boundary is protected by a synthetic test with events on `A`, `A+1`,
`A+30` and `A+31`.

## Memory and runtime strategy

The 30.6-million-row source is needed only during feature generation. Generated
features are cast to float32 and partitioned into compressed batches. Training
then restarts from the Parquet cache and reads only the compact 168-column view.
This keeps raw event data out of model memory and makes experiments resumable.

## Model and metric alignment

RMSLE is RMSE in `log1p(target)` space. Both LightGBM and XGBoost therefore use
squared-error regression directly on the transformed target. Predictions are
clipped at zero in log space and converted with `expm1` only when writing a
submission.

## Conservative ensembling

Candidate models are joined by `(sample_order, user_id, anchor_date)`. A single
nonnegative blend weight is fitted across the same four OOF folds. A diversity
candidate advances only when it is better standalone or improves the blend by a
pre-registered margin without an unsafe fold regression.
