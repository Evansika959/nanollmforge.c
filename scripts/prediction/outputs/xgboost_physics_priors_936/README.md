# XGBoost physical-prior feature ablation

Validation-selected variant: **kernel**. Fixed 598/150/188 train/validation/test split, three training seeds, four hyperparameter candidates per variant. Baseline metrics reproduce the preceding experiment.

| Variant | Target | Features | MAPE (%) | MAE | R² |
|---|---|---:|---:|---:|---:|
| baseline | tpot_ms | 14 | 23.39 ± 0.07 | 12.984 | 0.780 |
| baseline | ttft_ms | 14 | 19.71 ± 0.19 | 200.482 | 0.741 |
| baseline | dynamic_energy_per_token_mj | 14 | 28.07 ± 0.18 | 6.271 | 0.761 |
| compute | tpot_ms | 25 | 23.62 ± 0.05 | 12.903 | 0.786 |
| compute | ttft_ms | 25 | 19.31 ± 0.07 | 199.551 | 0.731 |
| compute | dynamic_energy_per_token_mj | 25 | 27.83 ± 0.15 | 6.170 | 0.762 |
| memory | tpot_ms | 26 | 23.80 ± 0.07 | 13.085 | 0.780 |
| memory | ttft_ms | 26 | 19.84 ± 0.17 | 204.280 | 0.719 |
| memory | dynamic_energy_per_token_mj | 26 | 28.38 ± 0.25 | 6.383 | 0.751 |
| kernel | tpot_ms | 21 | 23.51 ± 0.05 | 13.142 | 0.773 |
| kernel | ttft_ms | 21 | 19.59 ± 0.10 | 200.841 | 0.728 |
| kernel | dynamic_energy_per_token_mj | 21 | 29.32 ± 0.30 | 6.615 | 0.735 |
| all_physics | tpot_ms | 44 | 23.69 ± 0.07 | 13.051 | 0.778 |
| all_physics | ttft_ms | 44 | 19.86 ± 0.06 | 203.952 | 0.720 |
| all_physics | dynamic_energy_per_token_mj | 44 | 28.38 ± 0.16 | 6.355 | 0.756 |

## Feature definitions

- Baseline: the original 14 architecture/derived features.
- Compute: analytical MAC counts for projections, MLP, classifier and causal attention; prefill/decode separation, output-element count, and compute fractions.
- Memory: INT8 weights plus FP32 group scales, dequantized embedding storage, KV logical reads/writes/allocation, layer footprints and MAC/byte ratios.
- Kernel: MACs interacting with GS=16/32 specialized paths versus GS=64 general loop, number of quantization groups, approximate OpenMP region count and GQA reuse.
- All physics: union of the preceding features. This is feature engineering, not a physics-constrained loss or calibrated roofline time model.

## Assumptions and limitations

Fixed vocabulary 50,257, prompt 49, 31 decode forward calls (normally 32 output tokens), context capacity 256. MAC counts exclude normalization, nonlinear activation and sampling. Bytes describe logical footprints/accesses, not measured memory traffic. Prefill weight reuse and grouped KV cache reuse depend on the implementation and hardware. Kernel path features do not assert that compiler auto-vectorization is absent.
Only architecture/workload inputs are used. No measured performance, temperature, frequency, voltage, config ID or run order is a feature. Correlated feature gain is descriptive, not causal importance.
The existing test set has already been inspected. This is exploratory ablation, not an untouched confirmatory experiment. Bootstrap intervals condition on this split, are not corrected for multiple comparisons, and do not establish cross-session generalization. Seed SD is not uncertainty calibration.
Checkpoints are trained on 598 rows with validation early stopping; all 936 rows participate in the experiment but are not all training rows.

## Reproduce

```bash
conda activate nanollmforge
python scripts/sweep/compare_physics_priors.py --output scripts/sweep/outputs/xgboost_physics_priors_repeat
```
