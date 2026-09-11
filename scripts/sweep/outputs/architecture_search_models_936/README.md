# Architecture-only search models

All models use the original 598 training / 150 validation / 188 test architectures. Hyperparameters are chosen by three-fold CV inside training, with a separate early-stopping subset per fold. Final fitting uses the original validation rows only for early stopping. Test results are exploratory because this set was inspected in earlier experiments.

| Model | Target | MAPE % | Spearman | Recall of best 10% | Best-of-10 regret % |
|---|---|---:|---:|---:|---:|
| baseline | tpot_ms | 23.39 | 0.817 | 49.1% | 9.30 |
| baseline | ttft_ms | 19.71 | 0.867 | 68.4% | 1.45 |
| baseline | dynamic_energy_per_token_mj | 28.07 | 0.860 | 59.6% | 0.00 |
| group_experts | tpot_ms | 23.92 | 0.806 | 50.9% | 4.65 |
| group_experts | ttft_ms | 19.72 | 0.869 | 66.7% | 1.45 |
| group_experts | dynamic_energy_per_token_mj | 29.84 | 0.846 | 56.1% | 0.00 |
| physics_ridge | tpot_ms | 23.18 | 0.831 | 47.4% | 0.00 |
| physics_ridge | ttft_ms | 19.05 | 0.881 | 68.4% | 0.00 |
| physics_ridge | dynamic_energy_per_token_mj | 27.37 | 0.867 | 47.4% | 0.00 |
| physics_residual | tpot_ms | 23.23 | 0.830 | 47.4% | 0.00 |
| physics_residual | ttft_ms | 19.12 | 0.881 | 68.4% | 0.00 |
| physics_residual | dynamic_energy_per_token_mj | 27.08 | 0.868 | 52.6% | 0.00 |
| ranker | tpot_ms | n/a | 0.519 | 38.6% | 15.75 |
| ranker | ttft_ms | n/a | 0.693 | 45.6% | 3.96 |
| ranker | dynamic_energy_per_token_mj | n/a | 0.586 | 40.4% | 5.84 |

## Definitions and restrictions

- Baseline: original XGBoost log-target regressors. Group experts fit separate models for GS=16, 32, 64.
- Physics ridge: regularized linear model of log physical workload; physics residual adds an XGBoost residual model. No measured bandwidth or cache capacity is assumed.
- Ranker: XGBoost pairwise ranking with training-defined relevance deciles. Lower reported scores are better. Scores have no physical units; MAE/MAPE/R² are intentionally unavailable.
- Best 10% means 19 of 188 held-out candidates. Recall is overlap with the 19 truly best measured candidates. Best-of-10 regret compares the best actual result among 10 recommended candidates with the measured optimum in this test set.
- Means over three training seeds. Evaluation uses noisy single-run labels; identifying the recorded optimum is not proof of the true repeatable optimum.
- Inputs exclude temperatures, voltages, power measurements, performance measurements, run order and config ID. Labels enter only training or evaluation.
- Joblib checkpoints contain trusted local Python objects. They remain trained on the training partition, not all 936 examples.

## Reproduce

```bash
conda activate nanollmforge
python scripts/sweep/compare_search_surrogates.py --output scripts/sweep/outputs/architecture_search_models_repeat
```
