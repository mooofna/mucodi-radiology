# MuCoDi (radiology)

**Multi-teacher contrastive knowledge distillation for 3D radiology foundation models.**

MuCoDi distills the representations of several frozen radiology foundation-model
teachers into a compact 3D student encoder. It adapts the two-view contrastive
recipe of MoCo v3, replacing the momentum-encoder keys with cached, frozen teacher
embeddings drawn from a full-corpus feature bank, so a lightweight 3D-EfficientNet
student inherits the teachers' representations at a fraction of the parameters.

This is the code for the bachelor thesis *On the Parameter-Performance Trade-Off of
Radiology Foundation Models* (TU Dresden). It is a fork of the pathology companion
framework.

The repository has two coupled parts:

1. **The KD training framework:** `main.py`, `moco/`, `models/`, `dataprep/`, `utils/`.
2. **The `rate_eval` evaluation harness:** `rate-evals/`, a self-contained, tested
   package that loads each teacher through a wrapper and runs the read-out evaluations.

Orchestration is shared across both: `experiments/` (declarative studies and a Slurm
launcher), `jobs/` (env and Slurm settings), and `scripts/` (external-teacher setup).

This repository is the functional skeleton of the wider research codebase. It carries the
full training, evaluation, orchestration, and data-staging pipeline together with the three
thesis studies (the cross-cohort benchmark, the parameter-performance sweep, and the
CT-RATE to RAD-ChestCT deploy). To stay focused and runnable it leaves out the heavy and
exploratory material: raw data and run logs, generated figures and result tables, the
per-study analysis scripts and internal write-ups, and the exploratory studies (the
teacher-verification and reproduction campaign, the inference-cost benchmark, and the
single-node training variant). What remains runs end to end.

## Installation

Python 3.12 and a CUDA-capable GPU. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen   # installs the pinned lockfile exactly
# rate_eval needs `rave` (not on PyPI); --no-deps keeps the pinned torch pins:
uv pip install --no-deps "git+https://github.com/YalaLab/rave"
# the rate_eval harness ships its own pyproject; install it editable so its runtime deps
# (omegaconf, hydra, transformers, ...) and console scripts land in the venv:
uv pip install -e rate-evals   # rate-extract / rate-evaluate / rate-aggregate / rate-deploy
```

External teacher code and weights (TANGERINE, CT-CLIP, Pillar-0) are staged by
`scripts/setup_external_teachers.sh` (source `jobs/env.sh` first so `$HF_HOME` and
`$DATA_ROOT` are set). It also patches Pillar-0's Hub code, which ships without the
`self.post_init()` call its `trust_remote_code` load path needs.

## Data

The repository ships no datasets or labels; they are large and access-gated. The
`dataprep` stager regenerates the per-cohort manifests and label JSONs from the source
datasets (once you have access) into `$DATA_ROOT`:

```bash
python -m dataprep.datasets <cohort> --stage --label   # or --all
```

See each dataset's `SPEC` in `dataprep/datasets/` for its source and staging command.
Training reads CT-RATE V2; evaluation reads the held-out cohorts. The `$DATA_ROOT`,
`$RUNS_ROOT`, `$HF_HOME`, and related roots are exported by `jobs/env.sh`.

## Training

The student is trained by `main.py`, and the CLI defaults reproduce the thesis recipe:
AdamW at learning rate 1e-2, weight decay 1e-6, betas 0.9 and 0.95, temperature 0.2,
bfloat16 AMP, gradient clipping 1.0, and a frozen feature bank of M=16384 fp32 negatives
with study-level false-negative masking, run for 73 700 steps (3 685 warm-up) at seed 42.
A single-node run:

```bash
source jobs/env.sh
python main.py -a efficientnet3d_b0 \
  --multiprocessing-distributed --world-size 1 --rank 0 \
  --save-dir ./checkpoints
```

`--arch` selects the 3D-EfficientNet rung (registry in [`models/efficientnet3d.py`](models/efficientnet3d.py));
`--teacher-dims` and `--dataset` come from the profile in
[`dataprep/datasets.yaml`](dataprep/datasets.yaml). Training is step-based, distributed-only,
and W&B logging is optional.

The sweep holds a global batch of 32 (per-GPU batch 8 x 4 GPUs). The large rungs (b5/b6)
need ~80 GB GPUs at that per-GPU batch. On smaller GPUs, lower `--batch-size` and raise the
GPU count to keep the global batch at 32.

## Evaluation

The `rate_eval` harness runs three console-script stages (extract, cross-validate,
aggregate), plus `rate-deploy` for the cross-cohort deploy study:

```bash
rate-extract   --model curia2 --dataset lidc_chest_ct --all-splits --output-dir cache/curia2_lidc_chest_ct
rate-evaluate  cv --checkpoint-dir cache/curia2_lidc_chest_ct --labels-json <labels> --output-dir results/curia2_lidc_chest_ct --cv-folds 5
rate-aggregate cv-summaries --results-root results/ --teachers ... --out summary.csv
# cross-cohort deploy: train the head on CT-RATE, evaluate it on RAD-ChestCT, one abnormality class
rate-deploy    --source-checkpoint-dir cache/mucodi_student_ctrate --source-labels-json <ctrate-labels> \
               --target-checkpoint-dir cache/mucodi_student_radchestct --target-labels-json <radchestct-labels> \
               --class-name lung_nodule --output-dir results/mucodi_student_deploy
```

See [`rate-evals/README.md`](rate-evals/README.md) for the full harness tour and
[`rate-evals/CONTRACT.md`](rate-evals/CONTRACT.md) for the dataset/wrapper config contract.

## Running the studies

Each thesis experiment is a declarative study under `experiments/studies/`. The launcher
renders and submits its own Slurm job:

```bash
python -m experiments.launch experiments/studies/kd_student_sweep/kd_student_sweep.py --dry-run
python -m experiments.launch experiments/studies/kd_student_sweep/kd_student_sweep.py            # submit
python -m experiments.launch experiments/studies/kd_student_sweep/kd_student_sweep.py --continue # resume
```

The orchestration is cluster-agnostic: all Slurm settings (partition, account, QoS, paths)
are environment variables in [`jobs/env.sh`](jobs/env.sh). Edit its storage roots and Slurm
settings (the lines marked `EDIT`) once; no code changes are needed.

## Tests

```bash
pytest                    # root suites: moco/, models/, utils/, dataprep/, experiments/
cd rate-evals && pytest   # the rate_eval harness suite
```

## Reproducibility

Every rung is reproducible from the fixed seed 42 and the recipe above, given access to the
source datasets, the teacher model weights, and A100-class compute.

## Thesis-to-code map

| Thesis | Code |
|---|---|
| Ch 4 (Methods): distillation objective, feature bank, student | `moco/loss.py`, `moco/feature_bank.py`, `models/student.py`, `models/efficientnet3d.py`, `main.py`, `utils/engine.py` |
| Ch 5 (Evaluation Protocol): the `rate_eval` harness | `rate-evals/` |
| Sec 6.2 (cross-cohort transfer, incl. 6.2.1 student generalisation) | `experiments/studies/cross_cohort_benchmark/` |
| Sec 6.2.2: CT-RATE to RAD-ChestCT cold-deploy | `experiments/studies/ctrate_deploy_radchestct/` |
| Sec 6.3: parameter-performance trade-off | `experiments/studies/kd_student_sweep/` |

## License

Original code: **CC BY-NC 4.0** (see [`LICENSE`](LICENSE)). Third-party components
(MoCo v3, MONAI, the `rate_eval`/RAVE harness) are under their own licenses; see
[`NOTICE`](NOTICE). The bundled `rate-evals/` package carries its own license
([`rate-evals/LICENSE`](rate-evals/LICENSE), ECL 2.0).

## Citation

This code accompanies the bachelor thesis:

> Maurice Heide. *On the Parameter-Performance Trade-Off of Radiology Foundation
> Models.* Bachelor's thesis, TU Dresden, 2026.

Repository: <https://github.com/mooofna/mucodi-radiology>. Questions and issues: please
use the GitHub issue tracker.

Companion pathology study: Lenz et al., *Multi-Teacher Contrastive Distillation for
Edge-Efficient Pathology Foundation Models*, 2026 (arXiv:2607.05533, under review).
