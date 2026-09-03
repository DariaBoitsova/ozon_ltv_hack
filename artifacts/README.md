# Generated artifacts

This directory is reserved for reproducible outputs and contains no competition
data in the portfolio copy.

Each validation pipeline writes:

- one atomic Parquet checkpoint and JSON metric per fold;
- a combined keyed OOF Parquet;
- a summary JSON with parameters, versions and fold metrics.

Final CSVs are created only after exact row/order/finite/nonnegative checks and
are accompanied by SHA-256 manifests.
