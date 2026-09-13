# Hardware performance prediction

A reusable prediction package, separate from device measurement code in `scripts/sweep/`.

```text
scripts/prediction/
├── data/          Dataset versions, hardware CSV ingestion, frozen legacy snapshots
├── features/      Architecture-only analytic and hardware-aware features
├── models/        Neural definitions, XGBoost specification, checkpoint compatibility
├── training/      Shared optimizers, ensemble fitting, round-based training pipeline
├── inference/     Shared prediction API and CSV prediction commands
├── evaluation/    Metrics and historical-result audits
├── reporting/     Reports and plots; no model fitting
├── experiments/   Historical comparisons and fixed-cohort research experiments
├── tests/         Checkpoint regression, ingestion, split and leakage tests
└── outputs/       Existing models, snapshots, predictions and reports (unchanged)
```

Core modules do not import experiment scripts. Define a model in `models/`, fit it in `training/`, and expose its prediction in `inference/`. Shared architecture transformations belong in `features/`. Historical ablations retain their experiment-specific settings rather than silently changing old results.

## Commands

Run from the repository root in the existing `nanollmforge` environment:

```bash
python -m scripts.prediction --help
python -m scripts.prediction train --help
python -m scripts.prediction predict --help
python -m scripts.prediction dataset --help
```

The former flat `python scripts/prediction/<name>.py` commands are replaced by `python -m scripts.prediction <name-with-hyphens>`. For example:

```bash
python -m scripts.prediction compare-surrogates \
  --output scripts/prediction/outputs/surrogate_comparison_rerun

python -m scripts.prediction train-proposed-physics-surrogate \
  --output scripts/prediction/outputs/proposed_physics_surrogate_rerun

python -m scripts.prediction evaluate-gross-energy \
  --output scripts/prediction/outputs/gross_energy_comparison_1564_rerun
```

These commands reproduce the historical experiments. They are separate from the generic round-based `train` command below.

## Continual measurement workflow

The new round pipeline has no hard-coded sample count. It supports XGBoost and the grouped Transformer, with architecture-only physics32 inputs and log targets. Each round fits from scratch using all training observations in that dataset version. Stacked-ensemble research remains available through the historical experiment command.

### 1. Create the initial immutable dataset

Write a protocol JSON describing the **actual** acquisition conditions. Required fields are:

```json
{
  "protocol_id": "YOUR_VERIFIED_MEASUREMENT_PROTOCOL",
  "device_id": "YOUR_DEVICE_ID",
  "kernel_id": "YOUR_KERNEL_AND_BUILD_ID",
  "prompt_tokens": 49,
  "output_tokens": 32,
  "energy_target": "dynamic"
}
```

Replace the identity placeholders with verified identifiers. Do not reuse an identity across different kernel builds, devices or measurement methods. Add any other conditions that define comparability to the protocol object. Imports require an exact match of the whole object. A legacy CSV cannot prove its device/protocol identity; the operator must supply it correctly.

`energy_target` can be `dynamic` or `gross`. Both energy labels are prefill-inclusive per-output-token metrics. The current feature version deliberately supports only 49 actual prompt tokens and 32 outputs. Other workloads need a feature/protocol extension, not silent mixing.

To preserve the historical 1,564-architecture cohort and its 1,100/150/314 split:

```bash
python -m scripts.prediction dataset bootstrap \
  --previous scripts/prediction/outputs/batch2_progress_632 \
  --protocol /path/to/protocol.json \
  --output scripts/prediction/datasets/round_000.json
```

For another initial dataset, use the normalized schema documented below with explicit, architecture-disjoint split assignments.

### 2. Train and predict candidate architectures

```bash
python -m scripts.prediction train \
  --dataset scripts/prediction/datasets/round_000.json \
  --family xgboost \
  --output scripts/prediction/outputs/al_round_000

python -m scripts.prediction predict \
  --model scripts/prediction/outputs/al_round_000/model.joblib \
  --configs /path/to/candidate_architectures.csv \
  --output scripts/prediction/outputs/round_000_predictions.csv
```

Use `--family transformer` for the grouped Transformer. `--seed` selects one training seed; this command does not claim a multi-seed comparison. A run records its dataset snapshot/hash, protocol, model, split IDs, source hashes, library versions, held-out predictions and report.

Prediction column names follow the saved model's targets, including the distinction between gross and dynamic energy. Only trusted local joblib bundles should be loaded.

### 3. Measure selected candidates, then append a new round

Hardware measurements remain an explicit, separate step using the appropriate sweep protocol. Once a completed CSV batch has been reviewed:

```bash
python -m scripts.prediction dataset import-csv \
  --dataset scripts/prediction/datasets/round_000.json \
  --configs /path/to/measured_architectures.csv \
  --measurements /path/to/completed_measurements.csv \
  --protocol /path/to/protocol.json \
  --round-id round_001 \
  --output scripts/prediction/datasets/round_001.json

python -m scripts.prediction train \
  --dataset scripts/prediction/datasets/round_001.json \
  --family xgboost \
  --output scripts/prediction/outputs/al_round_001
```

The CSV adapter accepts the existing random-sweep field names and joins on `config_id`. It verifies the seven measured architecture dimensions against the configuration CSV. Gross energy is `total_energy_j * 1000 / output_tokens`; the stored dynamic-energy field is not recomputed. It refuses malformed or flagged rows rather than silently cleaning them. A different hardware output format needs an explicit adapter.

Measurement identity is `round_id` plus the supplied `measurement_id`/`run_id`, or otherwise `config_id:repeat`. Re-importing the same batch under the same round is rejected. Do not rename the round just to bypass duplicate detection. Repeated measurements require distinguishable repeat/run IDs. Source CSV hashes are retained.

### Split and provenance rules

- Dataset writes and run-directory creation are exclusive: existing versions are never overwritten.
- Appends preserve all earlier observations and record the parent dataset hash.
- New rows enter training only. An architecture already in validation/test cannot enter training, even under a new config ID.
- A config ID cannot refer to conflicting architectures. Duplicate measurement IDs are rejected.
- Nonpositive/nonfinite selected log targets and unsupported architecture shapes are rejected.
- All repeats of one architecture stay in one split. They are still separate, equally weighted observations; repeat aggregation and repeat-aware weighting are not implicitly chosen.
- Hardware calibration and neural scalers use training rows only. Validation controls early stopping. Measured power, temperatures, batch ID and acquisition order are never features.
- Repeatedly inspecting the same test across active-learning rounds remains exploratory. Keep a final prospective acquisition block untouched if you need an unbiased final assessment.

This provides the ingestion/retraining boundary for active learning. It does **not** yet choose candidates, calibrate uncertainty, schedule measurements, charge/cool the watch, or deploy a replacement predictor. Acquisition policy (uncertainty/diversity/Pareto search) can be added on top of the shared prediction API without changing model definitions.

## Python API and normalized batch schema

```python
from scripts.prediction.data.dataset import MeasurementDataset
from scripts.prediction.training.pipeline import fit_dataset
from scripts.prediction.inference.predictor import predict_bundle, bundle_targets
from scripts.prediction.models.serialization import load_bundle

dataset = MeasurementDataset.load("scripts/prediction/datasets/round_000.json")
next_dataset = dataset.append_training(batch)
next_dataset.save("scripts/prediction/datasets/round_001.json")
model = fit_dataset(next_dataset, family="xgboost", seed=42)
predictions = predict_bundle(model, candidate_configs)
targets = bundle_targets(model)
```

A normalized dataset has `schema_version: 1`, a `protocol` object, and an `observations` list. Each observation contains:

```text
measurement_id   unique measurement identity
config_id        architecture identity
round_id         acquisition round
split            train / validation / test
architecture     n_layer, d_model, n_h, n_kv, d_qk, d_v, d_mlp;
                 optional vocab_size (default 50257), q8_group_size (validated)
metrics          decode_tok_s, ttft_ms, and the selected energy metric:
                 dynamic_energy_per_token_mj or gross_energy_per_token_mj
```

A normalized append batch contains `protocol`, `observations`, and optional `source_hashes`. Its split may be omitted and is set to training. It can be imported with `dataset append --dataset ... --batch ... --output ...`.

## Historical artifacts and compatibility

All ten existing experiment directories remain in `outputs/`. Models, snapshots, measurements, reports and recorded historical hashes are unchanged. `relocation_manifest.json` describes the preceding move from `sweep/`; it is historical provenance, not a list of current module locations.

Old reports may show former commands/source paths. Use the new package CLI rather than editing those historical records. Trusted older checkpoints load through `models.serialization.load_bundle`, which supplies their old flat Python module names. Newly saved bundles use the package module paths.

Shared measurement/configuration files remain under `scripts/sweep/`. Stability-verification archives and inference C sources are untouched.

## Tests

```bash
python -m unittest discover -s scripts/prediction/tests -t . -v
```

Tests include historical checkpoint prediction equivalence, immutable rounds, protocol mismatches, duplicate IDs, architecture leakage, energy target labeling, CSV ingestion and a synthetic train/report run. The historical checkpoint tests require the existing local `outputs/` artifacts. Tests never benchmark the watch or retrain production checkpoints.
