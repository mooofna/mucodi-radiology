"""Radiology experiment configuration dataclasses (Study -> Experiment shape)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from experiments import cluster
from experiments.pipeline import RUNS_DIR


Protocol = Literal[
    "cv_5fold",
    "cv_5fold_macro",   # per-class loop, macro-mean
    "crossval_deploy",  # per-class (source,target) pairs; CV + deploy
]


@dataclass
class TrainingConfig:
    """KD training configuration; fields map to main.py flags."""
    arch: str = "efficientnet3d_b0"
    dataset_profile: str = "ctrate_kd"
    teacher_dims: str | None = None
    train_steps: int = 73_700
    warmup_steps: int = 3685              # absolute count, not fraction
    lr: float = 1e-2
    batch_size: int = 8                   # per-GPU
    weight_decay: float = 1e-6
    clip_grad: float = 1.0
    workers: int = 14
    save_every: int = 500
    num_layers: int = 1
    moco_t: float = 0.2
    amp: bool = True
    save_dir: str | None = None
    resume_from: str | None = None
    auto_resume: bool = False
    seed: int = 42
    num_gpus: int = 1
    num_nodes: int = 1
    cpus_per_task: int = field(default_factory=lambda: cluster.DEFAULT_TRAIN_CPUS)
    slurm_time: str | None = None
    grad_accum_steps: int = 1
    betas: str | None = None
    projector_arch: str | None = None     # "linear"/"mlp_dt"
    neg_source: str | None = None         # "gather" | "bank" | "bank-full"
    neg_bank_size: int | None = None
    neg_bank_dtype: str | None = None     # "float16" | "float32"
    neg_mask_false_negatives: str | None = None  # "off" | "self" | "study"
    env_vars: dict | None = None


@dataclass
class EvalConfig:
    """One (wrapper x dataset) evaluation cell; protocol selects the rate-evals entry."""
    wrapper: str
    dataset: str
    output_dir: str
    protocol: Protocol = "cv_5fold"
    cache_dir: str | None = None
    labels_json: str | None = None
    n_boot: int = 1000
    head_kind: str = "linear"
    slurm_time: str | None = None
    qa_key: str | None = None
    num_classes: int = 2                          # >2 for multi-class panels
    # cv --cohort patient-grouping key (ct_rate / radchestct / rspect / lidc)
    cohort: str | None = None
    # KD-suitability probe: L2-normalize features + weight_decay inner-CV
    l2_normalize: bool = False
    l2_grid: bool = False
    # cv_5fold_macro: per-class label JSONs to loop (17 radchestct / 18 ctrate)
    per_class_labels: list[str] | None = None
    # crossval_deploy: target cache + per-class labels (aligned 1:1) + grouping key
    cache_dir_target: str | None = None
    per_class_labels_target: list[str] | None = None
    cohort_target: str | None = None
    checkpoint_path: str | None = None
    model_arch: str | None = None
    # rate-extract resource overrides (default B=16 OOMs the 3D input)
    extract_batch_size: int = 2
    extract_num_gpus: int = 4
    extract_num_workers: int = 8


@dataclass
class Experiment:
    name: str
    training: TrainingConfig | None = None
    evaluations: list[EvalConfig] = field(default_factory=list)
    study: str | None = None

    def run_dir(self) -> Path:
        return RUNS_DIR / self.study / self.name if self.study else RUNS_DIR / self.name

    def checkpoint_file(self) -> Path | None:
        """Expected final-training checkpoint, or None if no training in this experiment."""
        if self.training is None:
            return None
        if self.training.save_dir:
            return Path(self.training.save_dir) / f"step_{self.training.train_steps:07d}.pth.tar"
        return self.run_dir() / "outputs" / "checkpoints" / f"step_{self.training.train_steps:07d}.pth.tar"

    def save(self, path: Path | None = None) -> Path:
        target = path or (self.run_dir() / "config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2))
        return target


def cv_5fold_eval(
    *,
    wrapper: str,
    dataset: str,
    cache_dir: str | Path,
    labels_json: str | Path,
    output_dir: str | Path,
    qa_key: str | None = None,
    num_classes: int = 2,
    n_boot: int = 1000,
    cohort: str | None = None,
    l2_normalize: bool = False,
    l2_grid: bool = False,
    head_kind: str = "linear",
    slurm_time: str | None = None,
    checkpoint_path: str | None = None,
    model_arch: str | None = None,
) -> EvalConfig:
    return EvalConfig(
        wrapper=wrapper,
        dataset=dataset,
        protocol="cv_5fold",
        cache_dir=str(cache_dir),
        labels_json=str(labels_json),
        output_dir=str(output_dir),
        qa_key=qa_key,
        num_classes=num_classes,
        n_boot=n_boot,
        cohort=cohort,
        l2_normalize=l2_normalize,
        l2_grid=l2_grid,
        head_kind=head_kind,
        slurm_time=slurm_time,
        checkpoint_path=checkpoint_path,
        model_arch=model_arch,
    )


def cv_5fold_macro_eval(
    *,
    wrapper: str,
    dataset: str,
    cache_dir: str | Path,
    per_class_labels: list[str | Path],
    output_dir: str | Path,
    n_boot: int = 1000,
    cohort: str | None = None,
    l2_normalize: bool = False,
    l2_grid: bool = False,
    head_kind: str = "linear",
    slurm_time: str | None = None,
    checkpoint_path: str | None = None,
    model_arch: str | None = None,
) -> EvalConfig:
    """Per-class CV evaluation with macro-mean aggregation across N classes (radchestct 17 / CT-RATE 18)."""
    return EvalConfig(
        wrapper=wrapper,
        dataset=dataset,
        protocol="cv_5fold_macro",
        cache_dir=str(cache_dir),
        per_class_labels=[str(p) for p in per_class_labels],
        output_dir=str(output_dir),
        n_boot=n_boot,
        cohort=cohort,
        l2_normalize=l2_normalize,
        l2_grid=l2_grid,
        head_kind=head_kind,
        slurm_time=slurm_time,
        checkpoint_path=checkpoint_path,
        model_arch=model_arch,
    )


def crossval_deploy_macro_eval(
    *,
    wrapper: str,
    dataset: str,
    cache_dir: str | Path,               # SOURCE cache
    per_class_labels: list[str | Path],  # SOURCE per-class labels
    cache_dir_target: str | Path,        # TARGET cache
    per_class_labels_target: list[str | Path],  # TARGET per-class labels
    output_dir: str | Path,
    n_boot: int = 1000,
    cohort: str | None = "ct_rate",
    cohort_target: str | None = "radchestct",
    l2_normalize: bool = True,
    head_kind: str = "mlp",
    slurm_time: str | None = None,
    checkpoint_path: str | None = None,
    model_arch: str | None = None,
) -> EvalConfig:
    """Cross-cohort crossval->deploy over N per-class (source, target) label pairs."""
    if len(per_class_labels) != len(per_class_labels_target):
        raise ValueError(
            f"crossval_deploy needs aligned source/target label lists; got "
            f"{len(per_class_labels)} source vs {len(per_class_labels_target)} target"
        )
    return EvalConfig(
        wrapper=wrapper,
        dataset=dataset,
        protocol="crossval_deploy",
        cache_dir=str(cache_dir),
        per_class_labels=[str(p) for p in per_class_labels],
        cache_dir_target=str(cache_dir_target),
        per_class_labels_target=[str(p) for p in per_class_labels_target],
        output_dir=str(output_dir),
        n_boot=n_boot,
        cohort=cohort,
        cohort_target=cohort_target,
        l2_normalize=l2_normalize,
        head_kind=head_kind,
        slurm_time=slurm_time,
        checkpoint_path=checkpoint_path,
        model_arch=model_arch,
    )
