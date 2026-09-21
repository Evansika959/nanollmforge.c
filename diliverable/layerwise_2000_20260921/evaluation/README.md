# XGBoost versus Transformer: layerwise measurements

Three-seed means on the same frozen test architectures. ± is training-seed SD. K=32.

| Model | Metric | Test N | MAPE (%) | Spearman | Kendall tau-b | Pairwise (%) | Recall@32 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | decode_tok_s | 200 | 26.06 ± 0.25 | 0.8235 | 0.6358 | 81.79 | 55.21 |
| XGBoost | ttft_ms | 200 | 21.95 ± 0.43 | 0.8899 | 0.7112 | 85.56 | 66.67 |
| XGBoost | dynamic_energy_per_token_mj | 200 | 11.57 ± 0.06 | 0.9528 | 0.8134 | 90.67 | 73.96 |
| Transformer-1x32 | decode_tok_s | 200 | 26.46 ± 0.16 | 0.8252 | 0.6393 | 81.97 | 61.46 |
| Transformer-1x32 | ttft_ms | 200 | 22.63 ± 0.29 | 0.8815 | 0.6972 | 84.86 | 63.54 |
| Transformer-1x32 | dynamic_energy_per_token_mj | 200 | 12.13 ± 0.14 | 0.9508 | 0.8097 | 90.49 | 71.88 |
| Transformer-2x64 | decode_tok_s | 200 | 26.52 ± 0.41 | 0.8172 | 0.6280 | 81.40 | 57.29 |
| Transformer-2x64 | ttft_ms | 200 | 22.73 ± 0.87 | 0.8834 | 0.6992 | 84.96 | 61.46 |
| Transformer-2x64 | dynamic_energy_per_token_mj | 200 | 11.78 ± 0.24 | 0.9528 | 0.8140 | 90.70 | 73.96 |

## Paired uncertainty for MAPE differences

Delta = Transformer − XGBoost, in percentage points. 3,000 paired bootstrap draws of whole layer-multiset groups.

| Transformer | Metric | Delta | 95% interval |
|---|---|---:|---:|
| Transformer-1x32 | decode_tok_s | 0.40 | [-0.84, 1.62] |
| Transformer-1x32 | ttft_ms | 0.68 | [-0.42, 1.71] |
| Transformer-1x32 | dynamic_energy_per_token_mj | 0.57 | [-0.18, 1.30] |
| Transformer-2x64 | decode_tok_s | 0.47 | [-0.69, 1.63] |
| Transformer-2x64 | ttft_ms | 0.78 | [-0.18, 1.77] |
| Transformer-2x64 | dynamic_energy_per_token_mj | 0.21 | [-0.55, 1.00] |
