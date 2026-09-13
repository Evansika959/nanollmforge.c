# Integrity checks and uncertainty

All integrity assertions passed; details are in audit.json.

Paired bootstrap resamples the 188 test architectures 10,000 times. Each architecture contributes its mean absolute percentage error across three seeds. Positive differences favor the proposed model. These are exploratory, unadjusted intervals on a previously inspected split, not fresh confirmatory evidence or learning-curve estimates.

| Model | Target | MAPE improvement (percentage points) | 95% interval |
|---|---|---:|---:|
| base_xgboost | decode_tok_s | -0.06 | [-0.78, 0.66] |
| base_xgboost | ttft_ms | 0.27 | [-0.44, 0.97] |
| base_xgboost | dynamic_energy_per_token_mj | -0.41 | [-1.54, 0.76] |
| grouped_transformer | decode_tok_s | -1.32 | [-2.59, -0.02] |
| grouped_transformer | ttft_ms | 0.05 | [-0.85, 0.99] |
| grouped_transformer | dynamic_energy_per_token_mj | -0.61 | [-2.25, 1.01] |
| stacked_ensemble | decode_tok_s | -0.26 | [-1.07, 0.55] |
| stacked_ensemble | ttft_ms | 0.46 | [-0.26, 1.20] |
| stacked_ensemble | dynamic_energy_per_token_mj | -0.56 | [-1.79, 0.71] |

## Interpretation

- No improvement interval is strictly positive. The stack does not demonstrate a reliable gain over the original XGBoost on this split.
- Training-layer weights span 1.28–7.77 MiB: all exceed the 512 KiB shared L2. Cached-layer bytes are constant, and spilled bytes are an affine transformation of layer bytes. This dataset does not identify a whole-layer cache-residency transition; this does not imply individual weight tiles cannot benefit from cache.
- The Transformer has 807,939 parameters and selects epochs 9, 9, and 16. This run alone cannot establish whether more samples will improve it or prove measurement noise is the limiting factor.
- Effective fitted bandwidth is about 1.92 GB/s, compute is 3.01 GMAC/s, per-layer decode overhead is 0.53 ms, and prefill overhead is 3.22 ms. These are regression coefficients, not independent hardware measurements.
- Keep the next acquisition batch untouched until the model and selection rules are fixed, then use it to test cross-batch generalization. Repeated anchor measurements would help distinguish environmental variation from architectural effects.
