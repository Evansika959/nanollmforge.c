# Layerwise hardware predictor — frozen 2,000-architecture release

Saved and trained on **2026-09-21**. This release is separate from the older
homogeneous active-learning bundle in the parent directory. No kernel edits,
hardware measurements, AL-state changes, or old-checkpoint replacements were made.

## Dataset

| Split | Architectures / throughput / TTFT labels | Valid dynamic-energy labels |
|---|---:|---:|
| Training | 1,600 | 1,599 |
| Validation | 200 | 200 |
| Test | 200 | 200 |
| Total | 2,000 | 1,999 |

Four completed registries contribute 500 accepted architectures each: fixed-KV v1,
variable-KV v2, and variable-KV v3 parts 1 and 2. Registry splits are preserved;
layer-multiset permutation groups never cross splits. There are 2,001 stored
measurement attempts but only 2,000 accepted rows. The accepted retest of
`LW_47d4353f9819966c231b` replaces the entire original metric row, not just its
energy label. Its superseded row remains in the SQLite backup and raw evidence.
The older training architecture `LW_e11e2a35f693c21d5fa0` still has no valid energy
label; its timing labels remain in training. No imputation or clipping to a fake
positive energy was used. Baseline-drift warnings are retained, not filtered.

The frozen dataset SHA-256 is
`e14673856a5f96c9a7d334f488b98dcbb1f0ca737f1e721c54900be6f63a73e4`.
All database labels were checked against raw result artifacts before fitting.

## Models and selection

All models receive the same **110 architecture-only features**, including global
quantization group size, parameter/weight footprints, KV-cache demand, per-layer
dimension statistics, adjacent-layer changes and quarter-of-depth summaries.
Temperature, battery, voltage, baseline power and measured timing are not inputs.
These summaries capture some order effects but do not uniquely encode every
heterogeneous layer sequence.

- **XGBoost:** three independent log-target regressors, depth 3, learning rate
  0.03, up to 1,500 trees per target, validation early stopping with patience 50.
- **Transformer 1×32:** one encoder layer, width 32, four attention heads,
  12,995 parameters.
- **Transformer 2×64:** two encoder layers, width 64, four attention heads,
  75,843 parameters.
- Both Transformers use 11 feature-group tokens plus CLS, joint three-target
  regression, train-only input/target scaling, masked log-space Smooth L1 loss,
  AdamW, cosine scheduling, and validation early stopping. Missing energy does
  not remove a row's timing supervision.
- Seeds: **42, 123, 2026**. Test arrays are not passed to fitting or selection.

The final artifact is **XGBoost, seed 2026**, at
[models/predictor_final.joblib](models/predictor_final.joblib).
Family selection minimizes the mean validation MAPE over targets and seeds;
seed selection minimizes the three-target validation MAPE within that family.
Family validation scores are 20.4833% (XGBoost), 20.9349% (Transformer 1×32), and
20.9195% (Transformer 2×64). No test-based selection was used.

The final checkpoint is the evaluated 1,600-row-training model, **not a refit on
all 2,000 rows**. Keeping 200 validation and 200 test architectures separate is
necessary to record its held-out performance honestly. Only 1,599 training rows
supervise its energy regressor.

### Final checkpoint: 200-test-architecture performance

| Metric | MAPE ↓ | Spearman ↑ | Kendall τ-b ↑ | Pairwise accuracy ↑ | Recall@32 ↑ |
|---|---:|---:|---:|---:|---:|
| Throughput | 25.91% | 0.8239 | 0.6381 | 81.90% | 56.25% |
| TTFT | 22.29% | 0.8881 | 0.7075 | 85.38% | 65.63% |
| Dynamic energy per output token | 11.61% | 0.9522 | 0.8127 | 90.63% | 75.00% |

MAPE is relative prediction error, not classification accuracy; do not interpret
100−MAPE as an accuracy rate. NDCG, pairwise accuracy for pairs separated by at
least 10%, and selection/top-32 regret are also saved in
[model_selection.json](evaluation/model_selection.json).

### Matched model comparison: three-seed test means

| Model | Throughput MAPE ↓ | TTFT MAPE ↓ | Dynamic-energy MAPE ↓ |
|---|---:|---:|---:|
| XGBoost | 26.06 ± 0.25% | 21.95 ± 0.43% | 11.57 ± 0.06% |
| Transformer 1×32 | 26.46 ± 0.16% | 22.63 ± 0.29% | 12.13 ± 0.14% |
| Transformer 2×64 | 26.52 ± 0.41% | 22.73 ± 0.87% | 11.78 ± 0.24% |

± is training-seed SD, not a confidence interval. XGBoost has the lowest mean
MAPE on each target, but all six paired group-bootstrap 95% intervals for
Transformer-minus-XGBoost differences include zero. This supports XGBoost as a
practical default, **not a statistically established universal advantage**.
The small Transformer is competitive, including slightly better mean throughput
ranking. Full rankings, per-seed scores, family/cohort breakdowns and paired
intervals are in [the comparison report](evaluation/README.md) and
[metrics.json](evaluation/metrics.json).

The previous 1,745-row comparison used 184 test rows, while this one uses 200.
Their aggregate errors cannot be interpreted as a fixed-test learning curve.
Previously reported test rows are reused, making this an exploratory held-out
evaluation rather than a new untouched confirmatory test.

## Protocol and units

Input: ordered heterogeneous architectures from the 135m/360m search families;
global effective quantization group size 16/32/64. INT8 weights and FP32 KV cache.
The kernel SHA-256 is
`16c954879607c83a1bb75538b883368d51bef732e340efe1c46a75c4f52362e6`.

Predictions, in column order:

1. `decode_tok_s`: decoding throughput, tokens/s.
2. `ttft_ms`: time to first token, milliseconds, for the actual 49-token prompt.
3. `dynamic_energy_per_token_mj`: prefill+decode integrated energy minus
   pre-idle median power × active duration, divided by 32 output tokens, in mJ.
   This is **not decode-only energy**.

Workload: 32 output tokens, 31 decode forwards, four CPU threads, CPU mask `f`,
100 ms power sampling, four-second pre/post idle windows. Temperature <45°C is
an admission condition, not a guarantee about peak inference temperature.
No temperature polling during inference. Full contracts are preserved.

## Contents and reproduction

- `dataset_snapshot.json`: all 2,000 architecture records, labels, grouped splits,
  warning metadata and provenance hashes.
- `databases/campaign1..4/`: consistent SQLite backups, including superseded rows,
  and hardware contracts.
- `measurement_evidence.zip`: trace/result/log/audit text files from all attempts
  and retests, under `campaign1..4/`; large reproducible synthetic weights omitted.
- `models/`: all nine fitted checkpoints plus the validation-selected final copy.
- `evaluation/`: exact training manifest, metrics, saved test predictions,
  histories, model-selection record and comparison report.
- `source_snapshot/`: training/reporting code and relevant shared dependencies.
- `data_manifest.json` and `release_hashes.json`: provenance and integrity hashes.

From the repository root with the recorded Python environment:

```bash
python -m scripts.prediction_research.layerwise_compare.run \
  --snapshot diliverable/layerwise_2000_20260921/dataset_snapshot.json \
  --output scripts/prediction/outputs/NEW_UNIQUE_2000_REPLAY
python -m unittest scripts.prediction_research.layerwise_compare.test_training \
  scripts.prediction_research.layerwise_refit.test_features
```

The current runner checks raw `artifact_path` values, which retain their original
absolute paths. A relocated replay must restore those artifacts from the evidence
archive or explicitly remap paths in a **new** snapshot, recording the changed
hash. Inference needs only the checkpoint and this repository's Python modules;
the release is not a standalone installed application. Exact package versions
are recorded in `evaluation/manifest.json`.

```python
import json
import joblib
from scripts.prediction_research.layerwise_refit.features import matrix
from scripts.prediction_research.layerwise_compare.training import predict

base = 'diliverable/layerwise_2000_20260921'
with open(base + '/dataset_snapshot.json') as f:
    data = json.load(f)
pack = joblib.load(base + '/models/predictor_final.joblib')
architectures = [data['rows'][0]['architecture']]  # replace with new architectures
x, _ = matrix(architectures, pack['features'])
print(pack['targets'])
print(predict(pack, x))
```

Only load trusted joblib files. For another release, `release freeze` requires a
new destination; `release package` verifies its dataset matches the completed
training run. Do not use this bundle as the older homogeneous AL workspace.
