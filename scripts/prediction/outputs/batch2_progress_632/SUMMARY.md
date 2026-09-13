# Does batch 2 improve prediction?

Frozen snapshot: 632 raw rows through RND_1632; 628 retained. Combined with the original 936: 1564 usable rows. Model fitting grows from 598 to 1100, with 150 fixed stopping rows and 314 fixed test rows.

Each arrow compares the same test architectures and the same model settings before/after adding training data. Values are mean MAPE across three training seeds, not ensemble-averaged predictions.

## New batch: 126 held-out architectures

| Model | Throughput MAPE | TTFT MAPE | Energy MAPE |
|---|---:|---:|---:|
| Original XGBoost (14 features) | 25.72% → 19.13% | 18.59% → 16.50% | 24.20% → 22.16% |
| Physics XGBoost (32 features) | 25.76% → 19.03% | 18.66% → 16.47% | 24.38% → 22.39% |
| Grouped Transformer | 25.57% → 21.68% | 18.66% → 17.64% | 23.90% → 23.21% |
| Stacking | 24.88% → 19.17% | 18.42% → 16.66% | 23.50% → 22.30% |

## Old batch: original188 held-out architectures

| Model | Throughput MAPE | TTFT MAPE | Energy MAPE |
|---|---:|---:|---:|
| Original XGBoost (14 features) | 23.84% → 21.92% | 19.71% → 20.77% | 28.07% → 29.73% |
| Physics XGBoost (32 features) | 23.91% → 21.85% | 19.44% → 20.80% | 28.47% → 29.77% |
| Grouped Transformer | 25.17% → 23.27% | 19.66% → 19.73% | 28.68% → 29.36% |
| Stacking | 24.11% → 22.12% | 19.25% → 20.27% | 28.63% → 29.13% |

## Pooled: all314 held-out architectures

| Model | Throughput MAPE | TTFT MAPE | Energy MAPE |
|---|---:|---:|---:|
| Original XGBoost (14 features) | 24.60% → 20.80% | 19.26% → 19.06% | 26.51% → 26.69% |
| Physics XGBoost (32 features) | 24.65% → 20.72% | 19.13% → 19.06% | 26.83% → 26.81% |
| Grouped Transformer | 25.33% → 22.63% | 19.26% → 18.89% | 26.76% → 26.90% |
| Stacking | 24.42% → 20.94% | 18.92% → 18.82% | 26.57% → 26.39% |

## Original XGBoost: data-growth curve

| Fitting rows | Old-test throughput | New-test throughput | New-test TTFT | New-test energy |
|---:|---:|---:|---:|---:|
| 598 | 23.84% | 25.72% | 18.59% | 24.20% |
| 849 | 22.30% | 21.41% | 17.44% | 23.12% |
| 1100 | 21.92% | 19.13% | 16.50% | 22.16% |
| 502 (batch2 only) | 20.37% | 12.95% | 14.13% | 22.23% |

## Paired uncertainty: added-data XGBoost14

| Test | Metric | Improvement in MAPE points | 95% interval |
|---|---|---:|---:|
| old_batch_test | decode_tok_s | 1.93 | [0.60, 3.29] |
| old_batch_test | ttft_ms | -1.06 | [-2.24, 0.13] |
| old_batch_test | dynamic_energy_per_token_mj | -1.67 | [-3.33, -0.11] |
| new_batch_test | decode_tok_s | 6.60 | [5.02, 8.13] |
| new_batch_test | ttft_ms | 2.09 | [0.69, 3.38] |
| new_batch_test | dynamic_energy_per_token_mj | 2.04 | [0.30, 3.72] |
| pooled_test | decode_tok_s | 3.80 | [2.76, 4.86] |
| pooled_test | ttft_ms | 0.20 | [-0.71, 1.09] |
| pooled_test | dynamic_energy_per_token_mj | -0.18 | [-1.39, 1.01] |

## Scope and checks

- Integrity checks passed: original 36 baseline metric values reproduced; source snapshot preserved; no architecture overlaps between fit/stop/test; all 15 OOF folds isolated; all meta weights nonnegative. New test IDs exclude batch2 rows inspected in the earlier state-signal experiment.
- Source CSVs were not edited. Zero/nonfinite targets are unsuitable for joint log training. The 36 s row RND_1250 is flagged under a 20 s gross-stall convention, not a proven failure diagnosis. Detailed README reports a separate stress-test including that row.
- Batch2 starts warmer (median 39.5°C versus 36.8°C). Temperature is diagnostic only, never an input. The batch2-only control helps expose distribution adaptation; it does not isolate a causal thermal effect.
- Bootstrap resamples test architectures and averages each row’s errors across seeds first. Intervals assume independent test rows and are not adjusted for many comparisons or temporal correlation. Previously reused old test remains exploratory.
- Preserve a future contiguous acquisition block for forward-transfer testing before claiming generalization to future watch sessions. The present new-batch split is random within the observed snapshot.
- Workload/labels unchanged: nominal48 prompt is inferred49 actual tokens,32 normal outputs, and dynamic energy includes prefill. No change to runq_reallm.c or device experiments.
