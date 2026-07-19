# rate-evals: radiology foundation-model benchmarking harness

Self-contained evaluation pipeline for radiology foundation models (CT-CLIP,
TANGERINE, Pillar-0, Curia-1/2). Lives inside the [MuCoDi-radiology](../) repo but ships its
own [pyproject.toml](pyproject.toml). Running `pip install -e .` here gets you the
`rate-extract`, `rate-evaluate`, `rate-aggregate`, and `rate-deploy` console scripts
independent of the parent project.

## Layered package

```
rate_eval/
+-- core/                primitives: errors, logging, device, seed, crossval (torch-free except device/seed)
+-- io/                  disk I/O: cache, cache_meta, download_lock, feature_loaders, result_schema
+-- config/              OmegaConf + Hydra loader + Pydantic dataset/preprocess schemas + CLI overrides
+-- pipelines/           pure-function extract orchestrator (extract, gpu_fanout, extract_loop)
+-- evaluation/          CV head training + metrics + plots
+-- datasets/            base classes + NiftiCTDataset + bespoke per-cell variants
+-- models/              teacher wrappers (Pillar0, TangerineVit, Curia2, CTClipZeroShot, ...) + TeacherWrapper Protocol
+-- cli/                 thin argparse shims that construct typed Request dataclasses + delegate to pipelines/
```

Key invariants kept by the layering:

- **`import rate_eval` is torch-free.** Enforced by [tests/contract/test_no_torch_at_top_level.py](tests/contract/test_no_torch_at_top_level.py). Torch-dependent helpers (e.g. `setup_device`) resolve via PEP-562 `__getattr__`.
- **Pipelines are pure functions** taking typed `ExtractRequest` / `EvaluateRequest` dataclasses, not argparse namespaces. CLI is the thin shim; `pipelines.extract.run(request)` can be called directly.
- **`TeacherWrapper` Protocol** (runtime-checkable) pins the wrapper contract; every entry in `components._MODEL_REGISTRY` is audited by [tests/contract/test_teacher_protocol.py](tests/contract/test_teacher_protocol.py).

## Installation

Installed as part of the parent project; see the root [README](../README.md). To install
just this harness into an already-activated parent venv:

```bash
pip install -e .          # this package + its console scripts
```

`rad-vision-engine` (RAVE, import `rve`) supplies the HU/MR windowing used by
`rate_eval/models/common.py`, and is a hard dependency in `pyproject.toml`, so
`pip install -e .` pulls it from git. For an editable RAVE checkout during development:

```bash
git clone https://github.com/YalaLab/rave ../rad-vision-engine && pip install -e ../rad-vision-engine
```

Console scripts land in `$VIRTUAL_ENV/bin/`:

```bash
rate-extract --help      # extract features into <cache>/embeddings/<split>/<accession>.npz
rate-evaluate cv --help  # CV evaluation on cached features
rate-aggregate --help    # cv-summaries
rate-deploy --help       # one cross-cohort crossval-to-deploy cell
```

## Typical workflows

### Extract features for one (wrapper x dataset) cell

```bash
rate-extract --model curia2 --dataset lidc_chest_ct --all-splits \
  --output-dir cache/curia2_lidc_chest_ct
```

CLI flags listed via `--help`. Hydra-style trailing overrides also work:
```bash
rate-extract --model pillar0_chest_ct --dataset lidc_chest_ct \
  hardware.batch_size_per_gpu=8 hardware.num_workers_per_gpu=4
```

Output:
- `cache/<wrapper>_<dataset>/embeddings/{train,valid,test}/<accession>.npz`: per-sample features
- `cache/<wrapper>_<dataset>/processed.csv`: append-only checkpoint log
- `cache/<wrapper>_<dataset>/cache_meta.yaml`: provenance sidecar (wrapper + dataset + preprocess + extraction params, sha256s)

The pipeline resumes automatically from `processed.csv` if interrupted.

### Evaluate cached features (CV mode)

```bash
rate-evaluate cv \
  --checkpoint-dir cache/curia2_lidc_chest_ct \
  --labels-json data/evaluation/lidc_malignancy_labels.json \
  --output-dir results/curia2_lidc_chest_ct \
  --cv-folds 5
```

Writes `results/<cell>/summary.json` conforming to [`rate_eval.io.result_schema.ResultSummary`](rate_eval/io/result_schema.py).

### Aggregate per-teacher results into a CSV

```bash
rate-aggregate cv-summaries \
  --results-root results/ \
  --teachers ctclip_zero_shot tangerine_vit curia2 \
  --suffix _lidc \
  --out results/teacher_cv_summary__lidc.csv
```

## Dataset configs (composition)

A cell is `(dataset x wrapper)`, composed from two axes; see [CONTRACT.md](CONTRACT.md)
for the full contract.

**Data-spec** `configs/dataset/<dataset>.yaml` (wrapper-agnostic, no `loader:` block):
```yaml
data:
  train_json: ${oc.env:REPO_ROOT}/data/.../train.jsonl
  valid_json: ${oc.env:REPO_ROOT}/data/.../dev.jsonl
  test_json:  ${oc.env:REPO_ROOT}/data/.../test.jsonl
img_paths_key: nii_path
modality: chest_ct
source: nifti              # nifti | npz   (optional)
```

**Wrapper default** `configs/model/<wrapper>.yaml` carries the preprocess applied to a
composed cell:
```yaml
default_loader_class: NiftiCTDataset
preprocess:
  pipeline: clip_interp_norm        # interp | passthrough | clip_interp_norm | interp_clip_norm
  target_shape: [240, 480, 480]
  hu_clip: [-1000.0, 1000.0]
  output_norm: halfrange_centered
```

Pipeline recipes (from [rate_eval/datasets/preprocess.py](rate_eval/datasets/preprocess.py)):
`interp` (Pillar-0 256^3), `passthrough` (Curia), `clip_interp_norm` (CT-CLIP),
`interp_clip_norm` (TANGERINE).

**Bespoke override** `configs/dataset/<dataset>/<wrapper>.yaml`: an explicit `loader:`
for combos with per-(dataset x wrapper) logic (CT-CLIP orientation, STOIC `.mha`,
CT-RATE HU-rescale):
```yaml
loader:
  class: RADChestCTForCTClip
data: { ... }
```

Resolution precedence (`rate_eval.components._get_loader_spec`): use the override
`<dataset>/<wrapper>.yaml` if present, otherwise compose
`<dataset>.yaml` with the wrapper default. The dataset class is resolved by `loader.class`
via [`components._load_dataset_class`](rate_eval/components.py). Adding a new dataset:
see [CONTRACT.md Sec 4](CONTRACT.md).

## Result schema

Two protocols emit `summary.json`:

| protocol         | typical use                              | required keys (in addition to common)                                 |
|------------------|------------------------------------------|----------------------------------------------------------------------|
| `cv`             | `rate-evaluate cv --cv-folds N`          | `cv_folds`, `cv_seed`, `pooled`, `pooled_auroc_ci`, `fold_summary`, `per_fold` |
| `single_split`   | legacy / forensic                        | (variant; see ResultSummary)                                          |

Common required keys: `schema_version`, `protocol`, `head_spec`, `feature_dim`,
optional `rate_eval_version`, `timestamp`, `cache_meta`.

The canonical Pydantic model is [`rate_eval.io.result_schema.ResultSummary`](rate_eval/io/result_schema.py). JSON is the source of truth; the CSVs are derived via `rate-aggregate`.

## Cache provenance

Every cache cell carries a [`cache_meta.yaml`](rate_eval/io/cache_meta.py) sidecar
recording: wrapper + dataset + preprocess + extraction params, with sha256s of the
dataset/wrapper YAMLs and the JSONL split manifests. On cache load,
[`load_features_from_cache(..., dataset_config_path=...)`](rate_eval/io/feature_loaders.py)
checks the recorded sha256 against the current YAML and emits a `WARN` on drift.

Set `RATE_SKIP_CACHE_META_CHECK=1` to suppress.

## Layout reference

```
rate-evals/
+-- rate_eval/                  # package (see "Layered package" above)
+-- configs/
|   +-- config.yaml             # Hydra base
|   +-- dataset/                # per-(dataset x wrapper) YAMLs
|   +-- model/                  # per-wrapper YAMLs (HF repo + checkpoint paths)
+-- data/                       # JSONL manifests + labels JSONs (gitignored)
+-- cache/                      # extracted features (gitignored)
+-- results/                    # CV summaries (gitignored)
+-- tests/
    +-- unit/                   # fast, pure
    +-- integration/            # end-to-end on synthetic data
    +-- contract/               # invariants every plugin must satisfy (no-torch import, TeacherWrapper)
```

## Test surface

```bash
pytest tests/ -v
```

| Suite | Files | Coverage |
|-------|------:|----------|
| `tests/unit/`        | 11 | stats, crossval, heads, preprocess, dataset_schema, result_schema, cache_meta |
| `tests/integration/` |  6 | NiftiCTDataset round-trip, aggregate CLI, dataset factory (over every dataset YAML), cohort-cluster CI, compare aligner, deploy |
| `tests/contract/`    |  2 | no-torch import + TeacherWrapper Protocol (parametrized over `_MODEL_REGISTRY`) |

The full suite runs ~190 passed / 10 skipped; skipped tests are optional-dep cells (rve/monai missing).

## Reproducibility

- Cached embeddings in `cache/*/embeddings/<split>/<accession>.npz` are deterministic and
  reproducible given fixed inputs.
- `rate-extract` is idempotent (resumes from `processed.csv`) and refuses to overwrite
  cached `.npz` files.

## Provenance

This package is a substantially-refactored derivative of YalaLab's RATE-Evals
(https://github.com/YalaLab/rate-evals), licensed ECL 2.0. Original (c) Yala Lab;
modifications (c) 2026 Maurice Heide. It depends at runtime on RAVE / rad-vision-engine
(https://github.com/YalaLab/rave) for HU/MR windowing.

## Citations

- Pillar-0: [Agrawal et al. 2025](https://github.com/YalaLab/pillar0)
- CT-CLIP: Hamamci et al., NeurIPS 2024
- TANGERINE: McConnell et al., 2026 (a computationally frugal chest-CT foundation model)
- Curia: Raidium Med, 2024
- MuCoDi: forthcoming
