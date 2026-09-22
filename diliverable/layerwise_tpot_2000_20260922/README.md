# XGBoost: TPOT / TTFT / dynamic energy

Dataset: 2000 architectures; split {'test': 200, 'train': 1600, 'validation': 200}.
110 architecture-only features; no temperature, battery, voltage, or measured timing inputs.
Three independent log-space XGBoost regressors per seed; maximum depth 3, up to 1500 trees,
learning rate 0.03, validation log-RMSE early stopping after 50 rounds.
Seeds: 42, 123, 2026. All three targets are lower-is-better.

## Test results: mean across three seeds

| Target | MAPE ↓ | Spearman ↑ | Kendall τ-b ↑ | Pairwise accuracy ↑ | Recall@32 ↑ |
|---|---:|---:|---:|---:|---:|
| tpot_ms | 15.50% | 0.8426 | 0.6666 | 83.33% | 97.92% |
| ttft_ms | 16.52% | 0.9225 | 0.7777 | 88.88% | 93.75% |
| dynamic_energy_per_token_mj | 13.97% | 0.9378 | 0.7919 | 89.59% | 62.50% |

## Saved checkpoint

`models/predictor_final.joblib`: seed 2026, selected ONLY by the
mean validation MAPE across the three targets, before test evaluation.
This is one seed, not an ensemble or a model refitted on validation/test rows.

| Target | Selected-checkpoint test MAPE |
|---|---:|
| tpot_ms | 15.46% |
| ttft_ms | 16.49% |
| dynamic_energy_per_token_mj | 13.97% |

## Definitions and limitations

- TPOT = 1000 × (end − prefill_end) / 31 ms/decode token, verified against raw timing and reciprocal throughput.
- TTFT uses the actual 49-token prompt (nominally 48); output count is 32.
- Dynamic energy includes prefill AND decode, baseline-subtracted and divided by 32 output tokens; it is not decode-only energy.
- All 417 baseline-drift warnings remain. All 2000 labels are positive and finite.
- GS64 uses the optimized kernel, GS16/32 retain earlier measurements; this is a mixed-kernel, non-contemporaneous dataset.
- Reused fixed test rows make this exploratory evaluation, not a fresh confirmatory holdout.
- Changing from throughput to reciprocal TPOT is not itself evidence of improved prediction. Their MAPE denominators differ.
- No new Transformer was trained in this release. Old throughput-based checkpoints and datasets are untouched.
- Target order and units are stored in every checkpoint; do not feed this checkpoint to old throughput-only reports/AL consumers.

## Reproduce

From the repository root, in the nanollmforge environment:

```bash
python -m scripts.prediction_research.layerwise_tpot.run \
  --snapshot diliverable/layerwise_2000_gs64_refresh_20260922/dataset_snapshot.json \
  --output scripts/prediction/outputs/layerwise_tpot_reproduction
```

The output directory must be new. `dataset_snapshot.json` is an exact copy of the source;
`labels.json` records the actual three training targets and preserved splits.
`test_predictions.npz`, `metrics.json`, `manifest.json`, and `source_snapshot/` record evaluation and provenance.

Inference (trusted local joblib files only):

```python
import joblib
from scripts.prediction_research.layerwise_refit.features import predict
bundle = joblib.load("diliverable/layerwise_tpot_2000_20260922/models/predictor_final.joblib")
values = predict(bundle, architectures)  # list of ordered-layer architecture dictionaries
print(bundle["targets"], bundle["units"], values)
```
