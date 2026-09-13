# Batch-2 data-growth evaluation

Snapshot: 632 raw batch-2 rows; 628 included; 3 invalid/nonpositive and 1 duration >20 s rows excluded from the main comparison. Full source snapshot and exact IDs retained. Original936 preserved.

Training grows 598 → 849 → 1100. Fixed original150 stopping rows, original188 test rows, and 126 new-batch test rows. Three training seeds, one split. Previously inspected early batch2 rows cannot enter the new test set. No temperature/state inputs or test-driven tuning.

| Test | Model | Training | N fit | Throughput MAPE | TTFT MAPE | Energy MAPE |
|---|---|---|---:|---:|---:|---:|
| old_batch_test | xgboost14 | old_only | 598 | 23.84% | 19.71% | 28.07% |
| old_batch_test | xgboost14 | plus_half | 849 | 22.30% | 20.04% | 28.53% |
| old_batch_test | xgboost14 | plus_all | 1100 | 21.92% | 20.77% | 29.73% |
| old_batch_test | xgboost14 | batch2_only | 502 | 20.37% | 23.07% | 33.59% |
| old_batch_test | xgboost32 | old_only | 598 | 23.91% | 19.44% | 28.47% |
| old_batch_test | xgboost32 | plus_half | 849 | 22.49% | 20.35% | 29.00% |
| old_batch_test | xgboost32 | plus_all | 1100 | 21.85% | 20.80% | 29.77% |
| old_batch_test | transformer32 | old_only | 598 | 25.17% | 19.66% | 28.68% |
| old_batch_test | transformer32 | plus_half | 849 | 24.23% | 19.51% | 28.40% |
| old_batch_test | transformer32 | plus_all | 1100 | 23.27% | 19.73% | 29.36% |
| old_batch_test | stack | old_only | 598 | 24.11% | 19.25% | 28.63% |
| old_batch_test | stack | plus_all | 1100 | 22.12% | 20.27% | 29.13% |
| new_batch_test | xgboost14 | old_only | 598 | 25.72% | 18.59% | 24.20% |
| new_batch_test | xgboost14 | plus_half | 849 | 21.41% | 17.44% | 23.12% |
| new_batch_test | xgboost14 | plus_all | 1100 | 19.13% | 16.50% | 22.16% |
| new_batch_test | xgboost14 | batch2_only | 502 | 12.95% | 14.13% | 22.23% |
| new_batch_test | xgboost32 | old_only | 598 | 25.76% | 18.66% | 24.38% |
| new_batch_test | xgboost32 | plus_half | 849 | 21.74% | 17.56% | 23.30% |
| new_batch_test | xgboost32 | plus_all | 1100 | 19.03% | 16.47% | 22.39% |
| new_batch_test | transformer32 | old_only | 598 | 25.57% | 18.66% | 23.90% |
| new_batch_test | transformer32 | plus_half | 849 | 23.39% | 18.19% | 23.70% |
| new_batch_test | transformer32 | plus_all | 1100 | 21.68% | 17.64% | 23.21% |
| new_batch_test | stack | old_only | 598 | 24.88% | 18.42% | 23.50% |
| new_batch_test | stack | plus_all | 1100 | 19.17% | 16.66% | 22.30% |
| pooled_test | xgboost14 | old_only | 598 | 24.60% | 19.26% | 26.51% |
| pooled_test | xgboost14 | plus_half | 849 | 21.95% | 19.00% | 26.36% |
| pooled_test | xgboost14 | plus_all | 1100 | 20.80% | 19.06% | 26.69% |
| pooled_test | xgboost14 | batch2_only | 502 | 17.39% | 19.48% | 29.03% |
| pooled_test | xgboost32 | old_only | 598 | 24.65% | 19.13% | 26.83% |
| pooled_test | xgboost32 | plus_half | 849 | 22.19% | 19.23% | 26.71% |
| pooled_test | xgboost32 | plus_all | 1100 | 20.72% | 19.06% | 26.81% |
| pooled_test | transformer32 | old_only | 598 | 25.33% | 19.26% | 26.76% |
| pooled_test | transformer32 | plus_half | 849 | 23.89% | 18.98% | 26.51% |
| pooled_test | transformer32 | plus_all | 1100 | 22.63% | 18.89% | 26.90% |
| pooled_test | stack | old_only | 598 | 24.42% | 18.92% | 26.57% |
| pooled_test | stack | plus_all | 1100 | 20.94% | 18.82% | 26.39% |
| new_test_plus_flagged_stalls | xgboost14 | old_only | 598 | 30.37% | 18.50% | 24.51% |
| new_test_plus_flagged_stalls | xgboost14 | plus_half | 849 | 25.48% | 17.41% | 23.41% |
| new_test_plus_flagged_stalls | xgboost14 | plus_all | 1100 | 23.14% | 16.48% | 22.44% |
| new_test_plus_flagged_stalls | xgboost14 | batch2_only | 502 | 16.77% | 14.12% | 22.49% |
| new_test_plus_flagged_stalls | xgboost32 | old_only | 598 | 30.38% | 18.55% | 24.70% |
| new_test_plus_flagged_stalls | xgboost32 | plus_half | 849 | 25.96% | 17.51% | 23.60% |
| new_test_plus_flagged_stalls | xgboost32 | plus_all | 1100 | 23.16% | 16.41% | 22.68% |
| new_test_plus_flagged_stalls | transformer32 | old_only | 598 | 29.99% | 18.54% | 24.18% |
| new_test_plus_flagged_stalls | transformer32 | plus_half | 849 | 27.23% | 18.08% | 23.97% |
| new_test_plus_flagged_stalls | transformer32 | plus_all | 1100 | 25.55% | 17.53% | 23.48% |
| new_test_plus_flagged_stalls | stack | old_only | 598 | 29.28% | 18.30% | 23.80% |
| new_test_plus_flagged_stalls | stack | plus_all | 1100 | 23.08% | 16.59% | 22.57% |

## Interpretation limits

- Paired bootstrap intervals in paired_bootstrap.json resample test architectures, using mean error across three seeds per architecture. Positive improvement favors added data. These are unadjusted exploratory intervals, not three independent data splits.
- Main filtering removes gross long-duration records under a declared 20s rule. This extends the previous removal of a 40s stall; it is not a hardware-derived failure criterion. new_test_plus_flagged_stalls reports predictions on excluded stalls, which never enter training.
- The original validation set remains fixed to isolate training-data additions. Its batch1-only distribution may limit adaptation to batch2.
- A batch2-only XGBoost control uses fewer fitting rows than the original model, with the same stopping set. Differences help separate acquisition-distribution effects from a simple sample-count story, but do not establish causality.
- Architecture counts, quantization and fixed workload match previous experiments. Nominal48 prompt implies49 actual tokens; dynamic energy includes prefill. No runtime C or measurement scripts were modified.
- Added-data models are evaluation checkpoints, not automatically deployed search predictors.

## Reproduce the exact snapshot

```bash
python scripts/sweep/evaluate_batch2_progress.py --snapshot scripts/sweep/outputs/batch2_progress_632/input_snapshot.json --output scripts/sweep/outputs/batch2_progress_repeat
```
