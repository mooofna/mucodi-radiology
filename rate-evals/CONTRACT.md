# rate-evals: the evaluation contract

`rate-evals` is a self-contained evaluation tool for radiology foundation models. Every
evaluation runs the same protocol, so results are comparable across teachers and datasets.
This document is that contract.

## 1. The unit of evaluation: a **cell = (dataset x wrapper)**

One evaluation cell pairs a **dataset** (what to evaluate on) with a **wrapper** (the
frozen teacher / student that produces features). Both are referenced by name; the
config system composes them.

- **Wrappers:** each is a `configs/model/<wrapper>.yaml` spec plus a `(module, class)` resolved
  from `rate_eval.components._MODEL_REGISTRY` (the module name need not match the slug). The
  registry holds the 6 teachers `pillar0_chest_ct, tangerine_vit, curia1, curia2,
  ctclip_zero_shot, ctclip_vocabfine_zero_shot`, the student `mucodi_student`, and a random-init
  floor (`random_features` / `random_features_s1`); every entry exposes
  `extract_features(inputs, modality) -> np.ndarray` (enforced by
  `tests/contract/test_teacher_protocol.py`).
- **Datasets** (`configs/dataset/`): see Sec 2.

## 2. Config = composition (orthogonal axes)

The dataset x wrapper space is composed, not enumerated:

```
configs/
  config.yaml                       # Hydra compose root (do not delete; defaults are seeds)
  model/<wrapper>.yaml              # wrapper spec + its DEFAULT loader: default_loader_class + preprocess
  dataset/<dataset>.yaml            # data-spec: data paths + modality + source (wrapper-agnostic)
  dataset/<dataset>/<wrapper>.yaml  # bespoke OVERRIDE: explicit loader (only where preprocessing is per-combo)
```

A cell's loader is resolved with this precedence (`rate_eval.components._get_loader_spec`):

1. **Override present:** `dataset/<dataset>/<wrapper>.yaml` supplies an explicit
   `loader:` block (a bespoke class such as `RADChestCTForCTClip` or
   `STOIC2021ForPillar0`, used where the loading/preprocessing is genuinely
   per-combination: orientation/spacing fixes, `.mha`, masked variants, HU rescale).
2. **Otherwise (the common case):** `dataset/<dataset>.yaml` (data only) is composed
   with the wrapper's `default_loader_class` + `preprocess` from `model/<wrapper>.yaml`.
   This is what collapses the per-(datasetxwrapper) explosion: the preprocessing is a
   property of the wrapper, applied uniformly across datasets.

Wrapper preprocessing is one recipe per wrapper: `interp` at 256^3 for Pillar-0,
`clip_interp_norm` at 240x480x480 for CT-CLIP, and `passthrough` for Curia. The one documented
exception is `radchestct_aniso`: a student-specific dataset variant (its own NPZ data-spec +
loader) that the student uses in place of sharing the teachers' `radchestct` cell.

## 3. The evaluation protocol

- **Extract** (`rate-extract --model <w> --dataset <d>`): turns one cell into cached frozen
  features under the cell's output dir, with a `cache_meta.yaml` provenance sidecar.
- **Provenance + idempotency**: `cache_meta.yaml` records the sha256 of *both* the
  dataset config and the wrapper config. Re-extraction is accession-idempotent; loading
  a cache whose config bytes drifted emits a WARN (`RATE_SKIP_CACHE_META_CHECK=1`
  suppresses). Never re-extract or re-evaluate a cell that already has artefacts unless you
  are deliberately regenerating them.
- **Evaluate** (`rate-evaluate cv --cv-folds 5 ...`): subject-level stratified k-fold CV head
  training on the frozen features (binary or multi-class). This is the canonical path.
- **CIs**: the **patient-cluster bootstrap** (resampling by patient group) is the
  canonical 95% CI. JSON (`ResultSummary`) is the source of truth; CSVs are derived via
  `rate-aggregate`.

`import rate_eval` is **torch-free** (pinned by `tests/contract/test_no_torch_at_top_level.py`);
torch is pulled lazily only when a model/dataset is actually constructed.

## 4. Adding a new evaluation dataset

The dataset *catalog* (public availability, staging, access protocol) lives in
`dataprep/datasets/registry.py` (parent repo), not here. To add a
cell to the benchmark:

1. **Stage + label** the dataset via dataprep: `python -m dataprep.datasets <name> --all`
   (parent repo; writes the `${DATA_ROOT}`-relative split manifests + label JSONs).
2. **Add a data-spec** `configs/dataset/<name>.yaml`: the `data:` block (train/valid/test
   JSON paths), `img_paths_key`, `modality`, and `source`/`npz_key`. No `loader:` block:
   it composes with each wrapper's default preprocess automatically.
3. **(Only if needed) add overrides** `configs/dataset/<name>/<wrapper>.yaml` for any
   wrapper whose loading/preprocessing for this dataset is bespoke (a non-default class).
4. Reference the cell from a study under `experiments/studies/` (parent repo; it passes
   `(dataset=<name>, wrapper=<w>)`; the resolver does the rest).

## 5. Package layout (the engine vs the cells)

```
rate_eval/
  components.py        # create_model / create_dataset factories + _MODEL_REGISTRY
  config/  io/  core/  pipelines/  cli/  evaluation/   # the harness engine
  models/*.py                                          # teacher + student wrapper classes (registry maps slug -> module.class)
  datasets/
    nifti.py preprocess.py readers.py ctclip_orient.py # consolidated loaders + helpers (engine; deep-imported)
    affine_repair.py                                   # kept top-level (imported by the training path)
    ctrate_metadata.py                                 # real-HU metadata helper for the ctrate loaders
    nifti_recipes.py                                   # the shared per-wrapper NIfTI recipes (NiftiFor{CTClip,Tangerine,Pillar0,Curia})
    _core/                                             # shared base classes (LIDCBaseDataset)
    <cohort>/for_<wrapper>.py                          # bespoke per-cohort loaders only where preprocessing is cohort-specific (ctrate, radchestct, stoic2021)
```

`create_dataset` resolves a `loader.class` string via `getattr(rate_eval.datasets, <class>)`,
so the only stable surface is the names re-exported in `datasets/__init__.py`; the cohort
subpackage layout is an implementation detail.
