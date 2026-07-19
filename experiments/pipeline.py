"""SBATCH templating primitives (JobSpec / header / preamble) + radiology job builders."""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from experiments import cluster

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "experiments" / "runs"
RATE_EVALS_DATASET_DIR = PROJECT_ROOT / "rate-evals" / "configs" / "dataset"


def _resolve_dataset_name(dataset: str, wrapper: str) -> str:
    """Resolve a (dataset, wrapper) cell to a Hydra-resolvable dataset name."""
    nested = f"{dataset}/{wrapper}"
    if (RATE_EVALS_DATASET_DIR / f"{nested}.yaml").exists():
        return nested
    if (RATE_EVALS_DATASET_DIR / f"{dataset}.yaml").exists():
        return dataset
    raise FileNotFoundError(
        f"No rate-evals dataset YAML found for dataset={dataset!r}, wrapper={wrapper!r}. "
        f"Searched: {nested}.yaml, {dataset}.yaml under {RATE_EVALS_DATASET_DIR}"
    )


SLURM_PARTITION = cluster.SLURM_PARTITION
SLURM_ACCOUNT = cluster.SLURM_ACCOUNT
SLURM_MAIL_USER = cluster.SLURM_MAIL_USER

_DEFAULT_CPUS_PER_TASK = cluster.DEFAULT_CPUS_PER_TASK
_DEFAULT_MEM = cluster.DEFAULT_MEM
# smaller mem so extract/eval jobs backfill onto RAM-tight GPU nodes
_EVAL_MEM = os.environ.get("MUCODI_EVAL_MEM", "64G")


# preamble sources jobs/env.sh (venv, cache/data contract, PYTHONPATH, HF offline)
_REPO_CODE = Path(__file__).resolve().parents[1]
ENV_PREAMBLE = textwrap.dedent(f"""\
    set -eo pipefail
    source {_REPO_CODE}/jobs/env.sh
    set -u
    cd "$REPO_ROOT"
    export PYTHONUNBUFFERED=1
""")


@dataclass(frozen=True)
class JobSpec:
    """Everything needed to emit one Slurm script."""
    name: str
    log_dir: Path
    body: str
    time: str = "06:00:00"
    mem: str = _DEFAULT_MEM
    cpus: int = _DEFAULT_CPUS_PER_TASK
    num_gpus: int = 1
    num_nodes: int = 1
    partition: str = SLURM_PARTITION


def _slurm_header(
    job_name: str,
    log_dir: Path,
    time: str = "06:00:00",
    mem: str = _DEFAULT_MEM,
    cpus: int = _DEFAULT_CPUS_PER_TASK,
    gpu: bool = True,
    num_gpus: int = 1,
    num_nodes: int = 1,
    partition: str = SLURM_PARTITION,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_dir / '%j_out.log'}",
        f"#SBATCH --error={log_dir / '%j_err.log'}",
        f"#SBATCH --time={time}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --nodes={num_nodes}",
        "#SBATCH --ntasks-per-node=1",
    ]
    if cluster.SLURM_CONSTRAINT:
        lines.append(f"#SBATCH --constraint={cluster.SLURM_CONSTRAINT}")
    if cluster.SLURM_EXCLUDE:
        lines.append(f"#SBATCH --exclude={cluster.SLURM_EXCLUDE}")
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    lines.append(f"#SBATCH --mem={mem}")
    if gpu and num_gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{num_gpus}")
    lines.extend([
        "#SBATCH --requeue",  # for auto_resume after preemption
        "#SBATCH --open-mode=append",
    ])
    if SLURM_ACCOUNT:
        lines.append(f"#SBATCH --account={SLURM_ACCOUNT}")
    if SLURM_MAIL_USER:
        lines.append("#SBATCH --mail-type=FAIL")
        lines.append(f"#SBATCH --mail-user={SLURM_MAIL_USER}")
    if cluster.SLURM_QOS:
        lines.append(f"#SBATCH --qos={cluster.SLURM_QOS}")
    return "\n".join(lines) + "\n"


def render_job(spec: JobSpec) -> str:
    """Turn a JobSpec into a complete Slurm script (header + env preamble + body)."""
    header = _slurm_header(
        spec.name, spec.log_dir,
        time=spec.time, mem=spec.mem, cpus=spec.cpus,
        gpu=(spec.num_gpus > 0), num_gpus=spec.num_gpus,
        num_nodes=spec.num_nodes, partition=spec.partition,
    )
    return header + "\n" + ENV_PREAMBLE + "\n" + spec.body


# lazy config imports avoid a circular dependency


def _job_train(exp) -> JobSpec:  # type: ignore[no-untyped-def]
    """KD training via ``python main.py`` (wall defaults to 08:00:00)."""
    from experiments.config import Experiment
    assert isinstance(exp, Experiment) and exp.training is not None

    t = exp.training
    save_dir = Path(t.save_dir) if t.save_dir else exp.run_dir() / "outputs" / "checkpoints"
    wall = t.slurm_time or "08:00:00"

    teacher_dims_arg = f' \\\n  --teacher-dims "{t.teacher_dims}"' if t.teacher_dims else ""
    resume_arg = f' \\\n  --resume "{t.resume_from}"' if t.resume_from else ""
    auto_resume_arg = " \\\n  --auto-resume" if getattr(t, "auto_resume", False) else ""
    amp_arg = " \\\n  --amp" if t.amp else ""
    grad_accum_arg = (
        f" \\\n  --grad-accum-steps {t.grad_accum_steps}" if getattr(t, "grad_accum_steps", 1) != 1 else ""
    )
    betas_arg = f' \\\n  --betas "{t.betas}"' if getattr(t, "betas", None) else ""
    projector_arg = (
        f" \\\n  --projector-arch {t.projector_arch}" if getattr(t, "projector_arch", None) else ""
    )
    # InfoNCE negative-source flags; all None -> main.py default.
    neg_source_arg = (
        f" \\\n  --neg-source {t.neg_source}" if getattr(t, "neg_source", None) else ""
    )
    neg_bank_size_arg = (
        f" \\\n  --neg-bank-size {t.neg_bank_size}"
        if getattr(t, "neg_bank_size", None) is not None else ""
    )
    neg_bank_dtype_arg = (
        f" \\\n  --neg-bank-dtype {t.neg_bank_dtype}" if getattr(t, "neg_bank_dtype", None) else ""
    )
    neg_mask_arg = (
        f" \\\n  --neg-mask-false-negatives {t.neg_mask_false_negatives}"
        if getattr(t, "neg_mask_false_negatives", None) else ""
    )

    env_lines = ""
    env_vars = getattr(t, "env_vars", None) or {}
    if env_vars:
        env_lines = "\n".join(f'export {k}="{v}"' for k, v in env_vars.items()) + "\n"

    num_nodes = getattr(t, "num_nodes", 1) or 1

    common_flags = textwrap.dedent(f"""\
          --arch {t.arch} \\
          --dataset {t.dataset_profile} \\
          --batch-size {t.batch_size} \\
          --lr {t.lr} \\
          --warmup-steps {t.warmup_steps} \\
          --moco-t {t.moco_t} \\
          --train-steps {t.train_steps} \\
          --save-every {t.save_every} \\
          --save-dir "{save_dir}" \\
          --weight-decay {t.weight_decay} \\
          --clip-grad {t.clip_grad} \\
          --num-layers {t.num_layers} \\
          --workers {t.workers} \\
          --seed {t.seed}{teacher_dims_arg}{amp_arg}{grad_accum_arg}{betas_arg}{projector_arg}{neg_source_arg}{neg_bank_size_arg}{neg_bank_dtype_arg}{neg_mask_arg}{resume_arg}{auto_resume_arg}""")

    body = textwrap.dedent(f"""\
            mkdir -p "{save_dir}"
            {env_lines}
            MASTER_PORT=$(shuf -i 10000-65000 -n 1)
            echo "Running on host: $(hostname)"
            nvidia-smi || true

            python main.py \\
{common_flags} \\
              --dist-url "tcp://localhost:$MASTER_PORT" \\
              --multiprocessing-distributed \\
              --world-size 1 \\
              --rank 0
        """)

    return JobSpec(
        name=f"train_{exp.name}",
        log_dir=exp.run_dir() / "logs",
        body=body,
        time=wall,
        num_gpus=t.num_gpus,
        num_nodes=num_nodes,
        cpus=getattr(t, "cpus_per_task", cluster.DEFAULT_TRAIN_CPUS),
    )


def _job_extract(exp, eval_cfg) -> JobSpec:  # type: ignore[no-untyped-def]
    """Extract embeddings via ``rate-extract`` (idempotent)."""
    cache_dir = Path(eval_cfg.cache_dir) if eval_cfg.cache_dir else None
    assert cache_dir is not None, "_job_extract requires eval_cfg.cache_dir"

    wall = eval_cfg.slurm_time or "04:00:00"
    ckpt_arg = (
        f' \\\n  model.checkpoint_path="{eval_cfg.checkpoint_path}"'
        if getattr(eval_cfg, "checkpoint_path", None)
        else ""
    )
    arch_arg = (
        f' \\\n  model.arch={eval_cfg.model_arch}'
        if getattr(eval_cfg, "model_arch", None)
        else ""
    )
    resolved_dataset = _resolve_dataset_name(eval_cfg.dataset, eval_cfg.wrapper)
    # Defaults (2, 4, 8) are safe for 3D students; rate-evals' default B=16 OOMs the big models.
    bs = getattr(eval_cfg, "extract_batch_size", 2)
    ng = getattr(eval_cfg, "extract_num_gpus", 4)
    nw = getattr(eval_cfg, "extract_num_workers", 8)
    # re-orient RAD-ChestCT NPZs (flipD_transHW) into the CT-RATE RAS frame
    orient_line = ("export RADCHESTCT_ORIENT=flipD_transHW  # A8 RAS-consistent RAD-ChestCT\n"
                   if str(eval_cfg.dataset).startswith("radchestct") else "")
    body = orient_line + textwrap.dedent(f"""\
        rate-extract \\
          --model {eval_cfg.wrapper} \\
          --dataset {resolved_dataset} \\
          --all-splits \\
          --batch-size {bs} \\
          --num-gpus {ng} \\
          --num-workers {nw} \\
          --output-dir "{cache_dir}"{ckpt_arg}{arch_arg}
    """)
    # checkpoint stem in dedup name -> distinct extracts per checkpoint
    _ckpt_tag = ("_" + Path(eval_cfg.checkpoint_path).stem) if getattr(eval_cfg, "checkpoint_path", None) else ""
    return JobSpec(
        name=f"extract_{exp.name}_{eval_cfg.wrapper}_{eval_cfg.dataset}{_ckpt_tag}",
        log_dir=exp.run_dir() / "logs",
        body=body,
        time=wall,
        mem=_EVAL_MEM,
        num_gpus=1,
    )


def _job_evaluate(exp, eval_cfg) -> JobSpec:  # type: ignore[no-untyped-def]
    """Emit the eval job: rate-evaluate (cv_5fold / cv_5fold_macro) or rate_eval.cli.deploy (crossval_deploy)."""
    p = eval_cfg.protocol
    out = eval_cfg.output_dir
    wall = eval_cfg.slurm_time

    if p == "cv_5fold":
        wall = wall or "02:00:00"
        qa_key_arg = f' \\\n  --qa-key "{eval_cfg.qa_key}"' if getattr(eval_cfg, "qa_key", None) else ""
        nc = getattr(eval_cfg, "num_classes", 2)
        num_classes_arg = f" \\\n  --num-classes {nc}" if nc != 2 else ""
        # --cohort groups multi-volume siblings into one fold for the patient-cluster BCa CI.
        cohort_arg = f' \\\n  --cohort "{eval_cfg.cohort}"' if getattr(eval_cfg, "cohort", None) else ""
        probe_arg = ""
        if getattr(eval_cfg, "l2_normalize", False):
            probe_arg += " \\\n  --l2-normalize"
        if getattr(eval_cfg, "l2_grid", False):
            probe_arg += " \\\n  --l2-grid"
        body = textwrap.dedent(f"""\
            rate-evaluate cv \\
              --checkpoint-dir "{eval_cfg.cache_dir}" \\
              --labels-json "{eval_cfg.labels_json}" \\
              --output-dir "{out}" \\
              --cv-folds 5 \\
              --head {eval_cfg.head_kind} \\
              --n-boot {eval_cfg.n_boot}{qa_key_arg}{num_classes_arg}{cohort_arg}{probe_arg}
        """)
    elif p == "cv_5fold_macro":
        assert getattr(eval_cfg, "per_class_labels", None), (
            "cv_5fold_macro requires per_class_labels (list of label JSONs)"
        )
        wall = wall or "03:00:00"
        labels_quoted = " ".join(f'"{p}"' for p in eval_cfg.per_class_labels)
        macro_summary_path = f"{out}/macro_summary.json"
        # --cohort lets the macro aggregator share one patient resample across classes
        cohort = getattr(eval_cfg, "cohort", None)
        cohort_arg = f' \\\n    --cohort "{cohort}"' if cohort else ""
        agg_cohort_arg = f' --cohort "{cohort}"' if cohort else ""
        if getattr(eval_cfg, "l2_normalize", False):
            cohort_arg += ' \\\n    --l2-normalize'
        if getattr(eval_cfg, "l2_grid", False):
            cohort_arg += ' \\\n    --l2-grid'
        body = textwrap.dedent(f"""\
            mkdir -p "{out}/per_class"
            FAILED_CLASSES=""
            for LABEL_JSON in {labels_quoted}; do
              CLASS_NAME=$(basename "$LABEL_JSON" .json | sed 's/^[^_]*_labels__//')
              CLASS_OUT="{out}/per_class/$CLASS_NAME"
              if [ -f "$CLASS_OUT/summary.json" ]; then
                echo "[skip] $CLASS_NAME (summary.json exists)"
                continue
              fi
              QA_KEY=$(python -c "import json; _d=json.load(open('$LABEL_JSON')); print(list(_d[next(iter(_d))]['qa_results']['default_qa'][0].keys())[0])")
              echo "[run] $CLASS_NAME qa_key=\\"$QA_KEY\\""
              mkdir -p "$CLASS_OUT"
              rate-evaluate cv \\
                --checkpoint-dir "{eval_cfg.cache_dir}" \\
                --labels-json "$LABEL_JSON" \\
                --qa-key "$QA_KEY" \\
                --output-dir "$CLASS_OUT" \\
                --cv-folds 5 \\
                --head {eval_cfg.head_kind} \\
                --n-boot {eval_cfg.n_boot}{cohort_arg} || FAILED_CLASSES="$FAILED_CLASSES $CLASS_NAME"
            done

            # Patient-clustered macro aggregation (one shared patient resample across all classes).
            python -m experiments.studies.macro_aggregate \\
              --per-class-dir "{out}/per_class" \\
              --out "{macro_summary_path}" \\
              --wrapper "{eval_cfg.wrapper}" \\
              --dataset "{eval_cfg.dataset}" \\
              --n-boot {eval_cfg.n_boot} \\
              --expected-classes {len(eval_cfg.per_class_labels)} \\
              --seed 42{agg_cohort_arg}
        """)
    elif p == "crossval_deploy":
        # per class: 5-fold source CV + fold-ensemble deploy to target, then macro_aggregate x2
        assert getattr(eval_cfg, "per_class_labels", None) and getattr(eval_cfg, "per_class_labels_target", None), (
            "crossval_deploy requires per_class_labels (source) + per_class_labels_target (target)"
        )
        assert len(eval_cfg.per_class_labels) == len(eval_cfg.per_class_labels_target), (
            "crossval_deploy source/target label lists must be aligned 1:1 by class"
        )
        assert getattr(eval_cfg, "cache_dir_target", None), "crossval_deploy requires cache_dir_target"
        wall = wall or "04:00:00"
        n_classes = len(eval_cfg.per_class_labels)
        src_arr = " ".join(f'"{s}"' for s in eval_cfg.per_class_labels)
        tgt_arr = " ".join(f'"{t}"' for t in eval_cfg.per_class_labels_target)
        src_cohort = eval_cfg.cohort or "ct_rate"
        tgt_cohort = getattr(eval_cfg, "cohort_target", None) or "radchestct"
        l2_arg = " \\\n                --l2-normalize" if getattr(eval_cfg, "l2_normalize", False) else ""
        body = textwrap.dedent(f"""\
            SRC=({src_arr})
            TGT=({tgt_arr})
            FAILED_CLASSES=""
            for i in "${{!SRC[@]}}"; do
              SRC_JSON="${{SRC[$i]}}"; TGT_JSON="${{TGT[$i]}}"
              CLASS_NAME=$(basename "$SRC_JSON" .json | sed 's/^[^_]*_labels__//')
              if [ -f "{out}/internal/per_class/$CLASS_NAME/summary.json" ] && [ -f "{out}/external/per_class/$CLASS_NAME/summary.json" ]; then
                echo "[skip] $CLASS_NAME (both summaries exist)"
                continue
              fi
              echo "[run] $CLASS_NAME"
              python -m rate_eval.cli.deploy \\
                --source-checkpoint-dir "{eval_cfg.cache_dir}" \\
                --source-labels-json "$SRC_JSON" \\
                --target-checkpoint-dir "{eval_cfg.cache_dir_target}" \\
                --target-labels-json "$TGT_JSON" \\
                --class-name "$CLASS_NAME" \\
                --output-dir "{out}" \\
                --head {eval_cfg.head_kind} \\
                --source-cohort "{src_cohort}" \\
                --target-cohort "{tgt_cohort}" \\
                --cv-folds 5 \\
                --n-boot {eval_cfg.n_boot}{l2_arg} || FAILED_CLASSES="$FAILED_CLASSES $CLASS_NAME"
            done
            [ -n "$FAILED_CLASSES" ] && echo "WARNING failed classes:$FAILED_CLASSES"

            # Internal (CT-RATE within-cohort CV) macro-AUROC/F1/AUPRC, patient-cluster BCa CI.
            python -m experiments.studies.macro_aggregate \\
              --per-class-dir "{out}/internal/per_class" \\
              --out "{out}/internal/macro_summary.json" \\
              --wrapper "{eval_cfg.wrapper}" --dataset "{eval_cfg.dataset}" \\
              --n-boot {eval_cfg.n_boot} --expected-classes {n_classes} --seed 42 --cohort "{src_cohort}"
            # External (CT-RATE->RAD-ChestCT deploy) macro-AUROC/F1/AUPRC, patient-cluster BCa CI.
            python -m experiments.studies.macro_aggregate \\
              --per-class-dir "{out}/external/per_class" \\
              --out "{out}/external/macro_summary.json" \\
              --wrapper "{eval_cfg.wrapper}" --dataset "{eval_cfg.dataset}" \\
              --n-boot {eval_cfg.n_boot} --expected-classes {n_classes} --seed 42 --cohort "{tgt_cohort}"
        """)
    else:
        raise ValueError(f"Unknown protocol: {p}")

    name = f"eval_{exp.name}_{eval_cfg.wrapper}_{eval_cfg.dataset}_{p}"
    # disambiguate the 2 crossval_deploy probe variants by output-dir leaf
    if p == "crossval_deploy":
        name = f"{name}_{Path(out).name}"
    return JobSpec(
        name=name,
        log_dir=exp.run_dir() / "logs",
        body=body,
        time=wall,
        mem=_EVAL_MEM,
        num_gpus=1,
    )


def _job_aggregate(exp) -> JobSpec | None:  # type: ignore[no-untyped-def]
    """Cross-evaluation aggregation, or None if nothing to aggregate."""
    cv_cells = [e for e in exp.evaluations if e.protocol == "cv_5fold"]

    if not cv_cells:
        return None

    lines: list[str] = []
    if cv_cells:
        # cv-summaries reads <results_root>/<entry>/summary.json
        run_dir = exp.run_dir()
        results_root = str(run_dir)
        entries = [str(Path(e.output_dir).relative_to(run_dir)) for e in cv_cells]
        out_csv = run_dir / "results_cv.csv"
        entries_arg = " ".join(entries)
        lines.append(textwrap.dedent(f"""\
            rate-aggregate cv-summaries \\
              --results-root "{results_root}" \\
              --teachers {entries_arg} \\
              --suffix "" \\
              --out "{out_csv}"
        """))
    body = "\n".join(lines)
    return JobSpec(
        name=f"aggregate_{exp.name}",
        log_dir=exp.run_dir() / "logs",
        body=body,
        time="00:30:00",
        num_gpus=0,
        cpus=2,
        mem="16G",
    )


@dataclass(frozen=True)
class ExperimentJobs:
    """All JobSpecs for one experiment."""
    train: JobSpec | None
    extracts: dict[str, JobSpec]   # cell key -> extract JobSpec
    evaluates: dict[str, JobSpec]  # cell key -> evaluate JobSpec
    aggregate: JobSpec | None


def build_experiment_jobs(exp) -> ExperimentJobs:  # type: ignore[no-untyped-def]
    """Compose JobSpecs for an Experiment, ready for launcher dependency wiring."""
    from experiments.config import Experiment
    assert isinstance(exp, Experiment)

    train = _job_train(exp) if exp.training is not None else None
    extracts: dict[str, JobSpec] = {}
    evaluates: dict[str, JobSpec] = {}
    # cells sharing a cache_dir share one extract JobSpec
    extract_by_cache: dict[str, JobSpec] = {}
    for ec in exp.evaluations:
        cell_key = ec.output_dir
        if ec.protocol in ("cv_5fold", "cv_5fold_macro"):
            spec = extract_by_cache.get(ec.cache_dir)
            if spec is None:
                spec = _job_extract(exp, ec)
                extract_by_cache[ec.cache_dir] = spec
            extracts[cell_key] = spec
        evaluates[cell_key] = _job_evaluate(exp, ec)
    aggregate = _job_aggregate(exp)
    return ExperimentJobs(train=train, extracts=extracts, evaluates=evaluates, aggregate=aggregate)
