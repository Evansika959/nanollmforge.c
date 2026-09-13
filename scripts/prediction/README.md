# Hardware performance prediction

Architecture-based surrogate training, inference, evaluation and reports live here, alongside `scripts/sweep/`.

```text
scripts/
├── sweep/
│   ├── run_*.py / run_*.bash          Device measurements
│   ├── generate_*.py                 Architecture sampling
│   ├── configs/                      Shared architecture configurations
│   └── outputs/*.csv                 Original and cleaned measurements
└── prediction/
    ├── compare_*.py / train_*.py      Predictor training
    ├── predict_*.py                   Saved-model inference
    ├── evaluate_*.py / audit_*.py     Evaluation and consistency checks
    ├── diagnose_*.py / report_*.py    Diagnostics and reports
    ├── physics_surrogate_components.py
    └── outputs/                      Checkpoints, snapshots, metrics and reports
```

## Running

From the repository root, activate the existing `nanollmforge` environment. Use a new output directory for training; the existing experiment directories are preserved.

```bash
# Original XGBoost / small Transformer comparison.
python scripts/prediction/compare_surrogates.py \
  --output scripts/prediction/outputs/surrogate_comparison_rerun

# Physics-feature grouped Transformer and stacked ensemble.
python scripts/prediction/train_proposed_physics_surrogate.py \
  --output scripts/prediction/outputs/proposed_physics_surrogate_rerun

# Paired gross-versus-dynamic energy experiment on the frozen 1564-row cohort.
python scripts/prediction/evaluate_gross_energy.py \
  --output scripts/prediction/outputs/gross_energy_comparison_1564_rerun

# Predict with an existing dynamic-energy stacked model (trusted local bundle).
python scripts/prediction/predict_proposed_physics_surrogate.py \
  --model scripts/prediction/outputs/proposed_physics_surrogate_936/stacked_ensemble_seed42.joblib \
  --configs scripts/sweep/configs/watch5_random_1000_50M_150M_sweep.csv \
  --output scripts/prediction/outputs/proposed_predictions.csv

# Read-only relocation regression checks; no device access or model training.
python -m unittest discover -s scripts/prediction -p 'test_prediction_layout.py' -v
```

The proposed-model inference command above is for the original dynamic-energy bundles. Do not use that entry point to label gross-energy checkpoints: it currently has fixed dynamic-energy output names. Gross checkpoints and their target metadata remain available through the gross-energy evaluation code.

## Existing experiments

- `outputs/surrogate_comparison_936/`: original XGBoost / Transformer comparison.
- `outputs/xgboost_physics_priors_936/`: physics-feature ablations.
- `outputs/architecture_search_models_936/`: architecture-only search predictors.
- `outputs/xgboost_error_diagnosis_936/` and `outputs/state_signal_investigation_936/`: error and environmental-state investigations.
- `outputs/transformer_learning_curve_936/`: sample-size experiment.
- `outputs/proposed_physics_surrogate_936/`: grouped Transformer / stacked ensemble.
- `outputs/batch2_progress_632/`: frozen combined-batch evaluation.
- `outputs/gross_energy_comparison_1564/`: gross-energy retraining and interpretation.
- `outputs/gross_energy_comparison_1564_setup_archive/`: preserved failed-setup metadata.

## Relocation and provenance

On 2026-09-13, 18 predictor scripts and 10 experiment directories moved from `scripts/sweep/` to this directory. Runtime defaults and generated command examples now use `scripts/prediction/`. Shared measurement/configuration paths remain under `scripts/sweep/`. No inference kernels or measurement protocols changed, and no models were retrained as part of the move.

Existing experiment artifacts were moved byte-for-byte, including model binaries, CSVs, JSON snapshots, reports and recorded source hashes. `relocation_manifest.json` records the checked artifact counts and hashes. Historical report commands, absolute paths and source hashes intentionally describe the original experiment, not the relocated/edited scripts. To follow an old command, change only the predictor script and model-result paths to `scripts/prediction/`; do not change measurement/configuration paths.

Top-level Python module names are unchanged so the trusted existing joblib bundles can still be loaded from these CLI entry points. Importing from another application requires this directory on its Python module search path, as the previous scripts required the sweep directory.
