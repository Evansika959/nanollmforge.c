# Proposed physics-informed surrogate experiment

936 architectures; original 598 fitting / 150 stopping / 188 test split; seeds 42, 123, 2026. All inputs are architecture-only. Results reuse a previously explored test set.

| Predictor | Target | MAPE % | MAE | R² |
|---|---|---:|---:|---:|
| original_xgboost | decode_tok_s | 23.84 ± 0.07 | 5.093 | 0.640 |
| original_xgboost | ttft_ms | 19.71 ± 0.19 | 200.482 | 0.741 |
| original_xgboost | dynamic_energy_per_token_mj | 28.07 ± 0.18 | 6.271 | 0.761 |
| grouped_transformer | decode_tok_s | 25.17 ± 0.22 | 5.277 | 0.628 |
| grouped_transformer | ttft_ms | 19.66 ± 0.28 | 201.785 | 0.731 |
| grouped_transformer | dynamic_energy_per_token_mj | 28.68 ± 0.77 | 6.571 | 0.747 |
| base_xgboost | decode_tok_s | 23.91 ± 0.14 | 5.098 | 0.638 |
| base_xgboost | ttft_ms | 19.44 ± 0.17 | 201.183 | 0.726 |
| base_xgboost | dynamic_energy_per_token_mj | 28.47 ± 0.30 | 6.415 | 0.759 |
| base_extratrees | decode_tok_s | 24.54 ± 0.08 | 5.247 | 0.601 |
| base_extratrees | ttft_ms | 19.47 ± 0.03 | 204.551 | 0.706 |
| base_extratrees | dynamic_energy_per_token_mj | 29.83 ± 0.10 | 6.789 | 0.714 |
| base_randomforest | decode_tok_s | 24.17 ± 0.10 | 5.156 | 0.620 |
| base_randomforest | ttft_ms | 19.42 ± 0.07 | 203.360 | 0.712 |
| base_randomforest | dynamic_energy_per_token_mj | 28.84 ± 0.17 | 6.654 | 0.733 |
| base_residual_mlp | decode_tok_s | 24.07 ± 0.33 | 5.061 | 0.639 |
| base_residual_mlp | ttft_ms | 19.36 ± 0.11 | 197.698 | 0.743 |
| base_residual_mlp | dynamic_energy_per_token_mj | 28.81 ± 0.06 | 6.585 | 0.731 |
| stacked_ensemble | decode_tok_s | 24.11 ± 0.28 | 5.132 | 0.636 |
| stacked_ensemble | ttft_ms | 19.25 ± 0.04 | 199.759 | 0.736 |
| stacked_ensemble | dynamic_energy_per_token_mj | 28.63 ± 0.29 | 6.532 | 0.752 |

## Implementation

- Exactly 32 documented features, grouped into seven disjoint semantic tokens. Transformer: CLS + learned positional embeddings, four encoder layers, eight heads, width 128, feedforward width 512, regression head.
- Four stacking bases: XGBoost, ExtraTrees, Random Forest, residual MLP. Five-fold OOF within the 598 fitting rows; each fold has its own calibration, normalization, and separate internal stopping rows. Nonnegative Ridge per target fits only the OOF predictions.
- Neural loss: Smooth L1 on training-standardized natural-log targets; AdamW with cosine schedule. Final outputs are exponentiated. Features are log1p transformed and train-standardized for neural models.
- Base-model OOF fits use fewer fitting rows than full bases because early-stopping rows are held separately. This can create stacking distribution shift; base predictions and weights are reported to expose it.

## Hardware and label provenance

- Google public specification: Snapdragon W5 Gen 2 Accelerated; Qualcomm public W5 Gen 2 brief: four A53 cores and LPDDR4 2133MHz. See sources in metadata.json.
- Connected Pixel Watch 5 reports shared L2 512KiB (CPUs 0–3) and L1 data cache 32KiB per core. Used as static cache-capacity prior.
- Effective bandwidth, compute MAC/s, synchronization and dispatch terms use nonnegative fits on training labels only. They are not published peak specifications or independently measured hardware constants.
- Cache residency/spill estimates are capacity proxies, not measured hit rates or DRAM traffic. Weight transfer and prefill compute formulas are roofline-inspired, not calibrated cycle-accurate simulation.
- Existing nominal 48-token measurements imply 49 actual prompt tokens and 31 decode forward calls, normally 32 output tokens. This experiment uses 49 in physical work estimates and preserves existing TTFT labels; it does not claim exact-48-token accuracy.
- Dynamic energy labels include prefill. They are not decode-only energy. The old baseline throughput is obtained by inverting its TPOT predictions, so its throughput MAPE differs from TPOT MAPE.
- No runq_reallm.c changes, no benchmark launches, no environmental inputs. All saved model predictions were reproduced after reload.

## Reproduce

```bash
conda activate nanollmforge
python scripts/sweep/train_proposed_physics_surrogate.py --output scripts/sweep/outputs/proposed_physics_surrogate_repeat
```

## Audit and architecture-only inference

See [AUDIT.md](AUDIT.md) for checks and paired bootstrap intervals. No proposed model demonstrates a reliable improvement over the original XGBoost on this split. Model bundles are trusted-local joblib files; do not load untrusted bundles.

```bash
python scripts/sweep/audit_proposed_physics_surrogate.py
python scripts/sweep/predict_proposed_physics_surrogate.py \
  --model scripts/sweep/outputs/proposed_physics_surrogate_936/stacked_ensemble_seed42.joblib \
  --configs scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv \
  --output scripts/sweep/outputs/proposed_predictions.csv
```

The Transformer bundle can be used with the same inference command. Predictions apply only to the fixed original sweep workload; extrapolation is unvalidated.
