"""Parameter-performance ladder: 3D-EfficientNet students distilled from Pillar-0 + Curia-1, linear-probed on RAD-ChestCT."""
from __future__ import annotations

import os
from pathlib import Path

from experiments.config import Experiment, TrainingConfig, cv_5fold_macro_eval
from experiments.studies.kd_student_sweep.ladder import ALL_RUNGS

STUDY_NAME = "kd_student_sweep"

REPO = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[3])
RUNS_OUT = REPO / "experiments" / "runs" / STUDY_NAME
DATA_EVAL = REPO / "data" / "evaluation"

DATASET_PROFILE = "ctrate_kd"
TEACHER_DIMS = "pillar0_chest_ct:1152,curia1:768"
WRAPPER = "mucodi_student"

# frozen bank supplies negatives; only global batch 32 is load-bearing
PER_GPU_B = 8
NUM_GPUS = 4
GLOBAL_BATCH = PER_GPU_B * NUM_GPUS
WALLTIME = os.environ.get("KDSWEEP_WALLTIME", "1-00:00:00")

N_TRAIN = 47149
N_EPOCHS = 50
STEPS_PER_EPOCH = -(-N_TRAIN // GLOBAL_BATCH)
FINAL_STEP = N_EPOCHS * STEPS_PER_EPOCH
WARMUP_FRAC = 0.05

_sel = os.environ.get("RUNGS")
RUNGS = [r.strip() for r in _sel.split(",")] if _sel else ALL_RUNGS
for r in RUNGS:
    if r not in ALL_RUNGS:
        raise ValueError(f"unknown rung {r!r}; valid: {ALL_RUNGS}")

# explicit set (not a glob): a stray label must not inflate the macro past 16
_RADCHESTCT16_SLUGS = [
    "atelectasis", "bronchiectasis", "calcification", "cardiomegaly", "consolidation",
    "emphysema", "hiatal_hernia", "interlobular_septal_thickening", "lung_nodule",
    "lung_opacity", "lymphadenopathy", "medical_material", "peribronchial_thickening",
    "pericardial_effusion", "pleural_effusion", "pulmonary_fibrotic_sequela",
]
RADCHESTCT_LABELS = sorted(str(DATA_EVAL / f"radchestct16_labels__{s}.json") for s in _RADCHESTCT16_SLUGS)
_missing = [q for q in RADCHESTCT_LABELS if not Path(q).is_file()]
if _missing:
    raise FileNotFoundError(
        f"missing radchestct16 paper-faithful labels: {_missing} -- run "
        "python -m dataprep.datasets radchestct --label first."
    )
assert len(RADCHESTCT_LABELS) == 16, f"expected 16 paper-faithful classes, got {len(RADCHESTCT_LABELS)}"


def _training(rung: str) -> TrainingConfig:
    return TrainingConfig(
        arch=rung,
        dataset_profile=DATASET_PROFILE,
        teacher_dims=TEACHER_DIMS,
        amp=True,
        betas="0.9,0.95",
        lr=1e-2,
        weight_decay=1e-6,
        moco_t=0.2,
        clip_grad=1.0,
        num_layers=1,
        projector_arch="linear",
        neg_source="bank",
        neg_bank_size=16384,
        neg_bank_dtype="float32",
        neg_mask_false_negatives="study",   # mask CT-RATE reconstruction-sibling false negatives
        batch_size=PER_GPU_B,
        grad_accum_steps=1,
        train_steps=FINAL_STEP,
        warmup_steps=round(WARMUP_FRAC * FINAL_STEP),
        save_every=500,
        auto_resume=True,
        seed=42,
        num_gpus=NUM_GPUS,
        num_nodes=1,
        slurm_time=WALLTIME,
        env_vars={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "WANDB_MODE": "online",
            "WANDB_NAME": f"{STUDY_NAME}|{rung}",
            "KDSWEEP_SKIP_CACHE_FILTER": "1",
        },
    )


def _radchestct_eval(rung: str, step: int, tag: str):
    """One cv_5fold_macro eval of rung at step, tagged (25ep/50ep) for distinct dirs."""
    ckpt = str(RUNS_OUT / rung / "outputs" / "checkpoints" / f"step_{step:07d}.pth.tar")
    ec = cv_5fold_macro_eval(
        wrapper=WRAPPER,
        dataset="radchestct_aniso",
        cache_dir=RUNS_OUT / rung / f"radchestct_{tag}" / "cache",
        per_class_labels=RADCHESTCT_LABELS,
        output_dir=RUNS_OUT / rung / f"radchestct_{tag}",
        cohort="radchestct",
        l2_normalize=True,
        l2_grid=True,
        slurm_time="08:00:00",
        checkpoint_path=ckpt,
        model_arch=rung,
    )
    # variable-size volumes can't be collated -> B=1
    ec.extract_batch_size = 1
    ec.extract_num_gpus = 1
    ec.extract_num_workers = 2
    return ec


# two within-rung reads: 25ep mid + 50ep final
EVAL_STEPS = [(FINAL_STEP // 2, "25ep"), (FINAL_STEP, "50ep")]

EXPERIMENTS: list[Experiment] = [
    Experiment(
        name=rung,
        training=_training(rung),
        evaluations=[_radchestct_eval(rung, step, tag) for (step, tag) in EVAL_STEPS],
    )
    for rung in RUNGS
]
