# Gross energy retraining on 1564 architectures

Same 1100 fitting / 150 early-stopping / 314 test architectures. Three seeds; mean of per-seed metrics, not ensemble predictions.
Gross energy = recorded total_energy_j × 1000 / 32. Includes prefill and device background. No runtime-state inputs.

| Test | Model | Target | Energy MAPE | MAE (mJ/token) | R² | Spearman | Within 10% |
|---|---|---|---:|---:|---:|---:|---:|
| old_batch_test | xgboost14 | dynamic | 29.73% | 6.23 | 0.780 | 0.867 | 12.6% |
| old_batch_test | xgboost14 | gross | 50.69% | 17.08 | 0.510 | 0.685 | 2.5% |
| old_batch_test | xgboost32 | dynamic | 29.77% | 6.28 | 0.777 | 0.864 | 11.0% |
| old_batch_test | xgboost32 | gross | 51.16% | 17.20 | 0.513 | 0.677 | 1.4% |
| old_batch_test | transformer32 | dynamic | 29.36% | 6.42 | 0.754 | 0.858 | 14.5% |
| old_batch_test | transformer32 | gross | 49.59% | 17.48 | 0.492 | 0.679 | 2.0% |
| new_batch_test | xgboost14 | dynamic | 22.16% | 5.96 | 0.811 | 0.923 | 24.3% |
| new_batch_test | xgboost14 | gross | 34.40% | 15.80 | 0.665 | 0.843 | 1.3% |
| new_batch_test | xgboost32 | dynamic | 22.39% | 6.17 | 0.797 | 0.921 | 22.5% |
| new_batch_test | xgboost32 | gross | 34.61% | 16.21 | 0.631 | 0.842 | 1.6% |
| new_batch_test | transformer32 | dynamic | 23.21% | 6.41 | 0.784 | 0.915 | 21.7% |
| new_batch_test | transformer32 | gross | 34.19% | 16.02 | 0.656 | 0.845 | 1.1% |
| pooled_test | xgboost14 | dynamic | 26.69% | 6.12 | 0.795 | 0.887 | 17.3% |
| pooled_test | xgboost14 | gross | 44.15% | 16.56 | 0.589 | 0.736 | 2.0% |
| pooled_test | xgboost32 | dynamic | 26.81% | 6.23 | 0.787 | 0.885 | 15.6% |
| pooled_test | xgboost32 | gross | 44.52% | 16.80 | 0.576 | 0.730 | 1.5% |
| pooled_test | transformer32 | dynamic | 26.90% | 6.42 | 0.769 | 0.878 | 17.4% |
| pooled_test | transformer32 | gross | 43.41% | 16.90 | 0.575 | 0.734 | 1.6% |

## Gross-target model metrics (pooled test)

| Model | Throughput MAPE | TTFT MAPE | Gross energy MAPE |
|---|---:|---:|---:|
| xgboost14 | 20.80% | 19.06% | 44.15% |
| xgboost32 | 20.72% | 19.06% | 44.52% |
| transformer32 | 22.19% | 18.77% | 43.41% |

## XGBoost14 gross-energy learning curve

| Fit rows | Old-test MAPE | New-test MAPE | Pooled MAPE |
|---:|---:|---:|---:|
| 598 | 44.03% | 37.03% | 41.22% |
| 849 | 47.84% | 35.21% | 42.77% |
| 1100 | 50.69% | 34.40% | 44.15% |

## Interpretation and checks

- Different physical target; lower percentage error is not itself lower absolute noise.
- Same previously inspected test set: exploratory, not a fresh prospective validation.
- Three seeds on one split, not three independent test sets.
- Keeps prior filtering including dynamic-positive selection and stall exclusion for paired comparison.
- Old measurement window/rounding limitations remain; removing baseline does not fix them.
- No stacking retraining in this focused XGBoost/Transformer comparison.
- Bootstrap compares errors on different physical energy targets. Positive delta favors gross. Resampling assumes independent architectures and does not account for temporal correlation.
- Reproduced 81 frozen baseline metrics; maximum MAPE discrepancy 0.00000149 percentage points. Gross and dynamic formulas reconcile within CSV rounding tolerance.
- Models are evaluation checkpoints, not automatically deployed. Original data, runq_reallm.c, and measurement scripts are unchanged.

## Reproduce

```bash
python scripts/sweep/evaluate_gross_energy.py --output scripts/sweep/outputs/gross_energy_comparison_1564_rerun
```
