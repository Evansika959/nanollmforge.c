# XGBoost vs small Transformer on 936 architectures

Architecture-only inputs. Fixed split: 598 training, 150 validation, 188 test architectures, stratified by Q8 group size. All 936 architectures are unique.
Both models have four validation-selected hyperparameter candidates, followed by three training seeds (42, 123, 2026) on the same split. Test data are used only for final evaluation.
XGBoost fits one model per target. The Transformer uses feature tokens, two attention layers, four heads, and a joint three-target head. Both learn log targets; Transformer feature/target normalization uses training rows only.

| Model | Target | MAE | MAPE (%) | R² | Spearman |
|---|---|---:|---:|---:|---:|
| xgboost | tpot_ms | 12.984 | 23.39 ± 0.07 | 0.780 | 0.817 |
| xgboost | ttft_ms | 200.482 | 19.71 ± 0.19 | 0.741 | 0.867 |
| xgboost | dynamic_energy_per_token_mj | 6.271 | 28.07 ± 0.18 | 0.761 | 0.860 |
| transformer | tpot_ms | 13.918 | 23.97 ± 0.66 | 0.715 | 0.829 |
| transformer | ttft_ms | 193.715 | 19.48 ± 0.24 | 0.753 | 0.879 |
| transformer | dynamic_energy_per_token_mj | 6.930 | 29.31 ± 0.94 | 0.679 | 0.859 |

MAPE ± sample SD across training seeds. These are not confidence intervals across independent train/test splits.

Paired test-row bootstrap (5,000 resamples) of Transformer minus XGBoost MAPE, averaging seed-specific errors per row. Positive values favor XGBoost:
- tpot_ms: 0.58 percentage points; conditional 95% interval [-1.00, 2.18].
- ttft_ms: -0.23 percentage points; conditional 95% interval [-1.32, 0.90].
- dynamic_energy_per_token_mj: 1.24 percentage points; conditional 95% interval [-0.59, 3.08].

## Interpretation limits

- One random held-out split supports within-distribution comparison, not generalization to unseen devices, kernels, workload lengths, or future measurement sessions.
- Random weights and uncontrolled measurement noise remain in the source data. No repeated measurements or session identifiers are available.
- Dynamic energy includes prefill and is divided by requested output tokens. It is not decode-only energy.
- No thermal, voltage, measured performance, config ID, or acquisition-order variables are model inputs.
- Seed dispersion is not calibrated epistemic uncertainty for active learning.
- Checkpoints are trained on the training partition only; validation controls early stopping. Do not mix these held-out evaluation results with a later all-data refit.

## Reproduce

```bash
conda activate nanollmforge
python scripts/sweep/compare_surrogates.py --output scripts/sweep/outputs/surrogate_comparison_repeat
python scripts/sweep/report_surrogates.py scripts/sweep/outputs/surrogate_comparison_repeat
```

Metadata stores input SHA256, exact split IDs, feature ordering, scalers, and library versions. Separate metrics, tuning scores, held-out predictions, selected hyperparameters, and checkpoints are included.
