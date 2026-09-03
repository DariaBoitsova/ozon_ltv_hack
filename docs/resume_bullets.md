# Resume-ready project description

## English

**Ozon Customer LTV Forecasting — Python, Polars, LightGBM, XGBoost, Parquet**

- Built a leakage-safe 30-day customer value forecasting pipeline over 30.6M
  daily marketplace activity records and 250k users, achieving 1.65408 public
  leaderboard RMSLE.
- Engineered 256 temporal, funnel, recency, cohort, gap and exponentially
  decayed behavioral features; reduced training to a non-redundant 168-feature
  float32 view cached in partitioned Zstandard Parquet.
- Designed rolling time validation with strict target-window eligibility,
  zero/nonzero segment diagnostics, synthetic boundary tests and resumable keyed
  OOF artifacts.
- Implemented metric-aligned log-target LightGBM/XGBoost pipelines and
  conservative OOF-only diversity blending with distribution, fold-safety and
  SHA-256 submission guards.

## Русский

**Прогнозирование LTV покупателей Ozon — Python, Polars, LightGBM, XGBoost,
Parquet**

- Построил защищённый от временных утечек pipeline прогноза 30-дневного LTV на
  30,6 млн дневных наблюдений и 250 тыс. пользователей; достиг public RMSLE
  1,65408.
- Разработал 256 временных, funnel, recency, cohort, gap и decay-признаков и
  сократил модельную матрицу до 168 неповторяющихся float32-признаков в
  партиционированном Zstandard Parquet.
- Реализовал rolling time-CV со строгой проверкой target windows, диагностикой
  zero/nonzero сегментов, синтетическими boundary-тестами и возобновляемыми OOF
  checkpoints.
- Собрал log-target LightGBM/XGBoost pipelines и безопасный OOF-ансамбль с
  контролем распределения, отдельных фолдов и SHA-256 артефактов.

Use only the bullets that match the role and be ready to explain why random
cross-validation and leaderboard-fitted calibration were rejected.
