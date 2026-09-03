# Data contract

The competition dataset is not redistributed in this portfolio directory.
Pipelines expect the following under the root passed as `--project-root`:

```text
data/
├── train.parquet       # daily user activity, immutable
└── sample_submit.csv   # 250,000 user IDs in required submission order
```

Expected source properties used by the original run:

- 30,631,006 rows and 18 columns;
- 250,000 unique users;
- one row per `(user_id, event_date)`;
- dates from 2025-01-01 through 2026-02-13;
- nonnegative activity and GMV columns;
- exact Search/Catalog identities for GMV, carts and orders.

Generated features are written to `fe/clean_v1/anchor_YYYYMMDD/batch_*.parquet`.
