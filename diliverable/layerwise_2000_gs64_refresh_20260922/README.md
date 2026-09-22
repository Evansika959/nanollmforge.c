# Latest 2,000-point layerwise dataset — GS64 refresh

Created **2026-09-22**. All 664 GS64 records are replaced with complete optimized
measurements; the other 1,336 records are retained. This is a **new version**.
Original databases, the 2026-09-21 data release and trained models are unchanged.
No predictor was retrained or promoted in this update.

## Key files

| File | Content |
|---|---|
| [dataset_snapshot.json](dataset_snapshot.json) | Authoritative 2,000 architectures, labels, splits and per-row kernel provenance |
| [dataset.sqlite](dataset.sqlite) | Same data in `observations`, with target columns and full `row_json` |
| [manifest.json](manifest.json) | Version hash, counts, protocol and descriptive statistics |
| [replacement_audit.json](replacement_audit.json) | 664 old/new label pairs, result/weight hashes, and 1,336 unchanged IDs |
| [plots/updated_2000_scatter.png](plots/updated_2000_scatter.png) | All updated points: parameter count versus throughput, TTFT and energy |
| [plots/gs64_paired_comparison.png](plots/gs64_paired_comparison.png) | Matched GS64 old/new measurements on logarithmic axes |
| `plots/*.pdf` | Vector equivalents of both plots |
| `evidence/results/` | Exact raw result JSON files for the selected 2,000 measurements |
| `measurement_evidence.zip` | Selected traces, timing, results and provenance, by architecture ID |
| `provenance/` | Original dataset, new registry backup, hardware contract, both C kernels and validation evidence |
| `source_snapshot/` | Merge/plot code, tests and supporting feature/measurement code |
| `verification.json`, `release_hashes.json` | Integrity checks and hashes of delivered files |

## Composition

| Split | Architectures | Valid throughput | Valid TTFT | Valid dynamic energy |
|---|---:|---:|---:|---:|
| Train | 1,600 | 1,600 | 1,600 | 1,600 |
| Validation | 200 | 200 | 200 | 200 |
| Test | 200 | 200 | 200 | 200 |
| Total | 2,000 | 2,000 | 2,000 | 2,000 |

GS16: **672**; GS32: **664**; GS64: **664**. Each model family has 1,000
architectures. The replacements preserve their **536/64/64** train/validation/test
assignments. All original row order, architecture IDs, ordered layers, cohorts
and permutation groups are unchanged. Permutation groups never cross splits.

The formerly missing energy label (`LW_e11e2a35f693c21d5fa0`) is now measured.
No imputation or artificial clipping was used. All **417 baseline-drift warnings**
are retained: 284 from new GS64 and 133 from retained GS16/32 measurements.
A valid positive energy label does not imply a noise-free measurement.

## Replacement and measurement protocol

Each GS64 row receives **all three targets from its new measurement**, together
with its new source, artifact hash, protocol hash and warning. Old timing and new
energy are never spliced. Derived fields include TPOT, decode duration and gross
energy per decode forward. All 664 new raw results were re-parsed from traces
and timestamps; database labels, model hashes, architecture identity and final
generated tokens were checked. The 1,336 retained rows keep their original labels.

This is explicitly a **mixed-kernel dataset**, not 2,000 measurements from one
kernel revision: GS16/32 retain historical measurements; GS64 uses the optimized
decode source. Each row records `measurement_revision`, `kernel_sha256` and
`protocol_sha256`. The top-level `protocol` contains the common workload and a
`kernel_by_group` map. GS16/32 code paths were not deliberately changed by this
specialization, but those groups were not remeasured. Compiler/session effects
cannot be assumed away when training or reporting from this dataset.

Common workload: actual 49-token prompt, 32 output tokens, 31 decode forwards,
four CPU threads, INT8 weights and FP32 KV cache. Admission temperature <45°C,
no temperature polling during inference, battery cutoff 30%.

Units: throughput **tokens/s**, TTFT **ms** (plots show seconds), dynamic energy
**mJ/output token**. Dynamic energy includes prefill+decode integrated energy,
minus pre-idle median power × active duration, divided by 32 outputs. It is not
decode-only energy.

## Plot interpretation

The full scatter uses color for GS and marker shape for family. The refreshed
GS64 points form a higher-throughput, lower-TTFT band; energy still shows more
scatter. Different GS strata contain different architectures, so cross-color
differences are not causal group-size comparisons on identical models.

Matched GS64 median changes are **3.00× throughput**, **33.9% lower TTFT**, and
**46.2% lower dynamic energy**. Energy has 663 pairs because one old label was
invalid. Medians of paired ratios differ from ratios of marginal medians.
New and historical measurements were collected at different times; their
differences include environmental variation. This is not a predictor-accuracy result.

## Use and reproduce

From the repository root in the `nanollmforge` environment:

```python
import json
from scripts.prediction_research.layerwise_refit.features import matrix
with open('diliverable/layerwise_2000_gs64_refresh_20260922/dataset_snapshot.json') as f:
    dataset = json.load(f)
x, names = matrix([r['architecture'] for r in dataset['rows']])
assert x.shape == (2000, 110)
```

Reproduce into a **new, nonexistent** directory:

```bash
python -m scripts.prediction_research.layerwise_refit.refresh_gs64 \
  --output diliverable/NEW_GS64_REFRESH_VERSION
python -m scripts.prediction_research.layerwise_refit.plot_refresh \
  --dataset-dir diliverable/NEW_GS64_REFRESH_VERSION
python -m unittest scripts.prediction_research.layerwise_refit.test_refresh_gs64
```

The publisher verifies the original release and complete replay, then refuses
to overwrite an existing destination. Re-running the plot command regenerates
only plot artifacts. New measurements require a new dataset version.

Future matched training can use this snapshot explicitly (**not run here**):

```bash
python -m scripts.prediction_research.layerwise_compare.run \
  --snapshot diliverable/layerwise_2000_gs64_refresh_20260922/dataset_snapshot.json \
  --output scripts/prediction/outputs/NEW_UNIQUE_REFRESH_COMPARISON
```

Without `--snapshot`, that runner still reads the original four unchanged
registries. Old checkpoints retain the old dataset fingerprint and must not be
described as trained on this new version.

Raw results are copied into this bundle. `artifact_path` is an absolute path for
the existing runner; `artifact_relative_path` supports relocation. After moving
the bundle, remap absolute paths in a **new** snapshot and record its new hash.
The original absolute source paths are provenance only.
