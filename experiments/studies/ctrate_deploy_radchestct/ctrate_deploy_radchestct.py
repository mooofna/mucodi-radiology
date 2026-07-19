"""5-fold CV a probe on CT-RATE, deploy the fold-heads to RAD-ChestCT (mean-of-probs ensemble)."""
import os
from pathlib import Path

from experiments.config import Experiment, crossval_deploy_macro_eval
from experiments.studies._layout import cohort_paths
from experiments.studies._roster import build_models

STUDY_NAME = "ctrate_deploy_radchestct"

# parents[3] = repo root
REPO = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[3])
DATA_EVAL = REPO / "data" / "evaluation"
RUNS_OUT = REPO / "experiments" / "runs" / STUDY_NAME
# read-only feature caches from the parent benchmark
RUNS_OUT_CCB = REPO / "experiments" / "runs" / "cross_cohort_benchmark"
KD_RUNS = REPO / "experiments" / "runs" / "kd_student_sweep"

# merged CT-RATE calcification source label
DERIVED_CALC = Path(__file__).resolve().parent / "results" / "derived_labels" / "ctrate_labels__calcification.json"


MODELS = build_models(KD_RUNS, "CDR_MODELS")


_MOSAIC = "mosaic_attenuation_pattern"  # 0 positives in RAD-ChestCT -> dropped


def _draelos_pairs() -> tuple[list[str], list[str]]:
    """Aligned (source, target) label paths for the 16-class Draelos panel."""
    target_paths = [
        p for p in sorted(DATA_EVAL.glob("radchestct16_labels__*.json")) if _MOSAIC not in p.name
    ]
    if not target_paths:
        raise FileNotFoundError(f"no radchestct16_labels__*.json under {DATA_EVAL}")
    src: list[str] = []
    tgt: list[str] = []
    for tp in target_paths:
        cls = tp.stem.split("_labels__", 1)[1]
        if cls == "calcification":
            if not DERIVED_CALC.exists():
                # auto-build merged calcification label (idempotent)
                from experiments.studies.ctrate_deploy_radchestct.build_calcification_label import build
                build()
            sp = DERIVED_CALC
        else:
            sp = DATA_EVAL / f"ctrate_labels__{cls}.json"
        if not Path(sp).exists():
            raise FileNotFoundError(
                f"source label for Draelos class {cls!r} not found: {sp} "
                f"(run build_calcification_label.py for calcification)"
            )
        src.append(str(sp))
        tgt.append(str(tp))
    if len(src) != 16:
        raise AssertionError(f"expected the 16-class Draelos panel, got {len(src)}: {[Path(t).name for t in tgt]}")
    return src, tgt


_SRC_LABELS, _TGT_LABELS = _draelos_pairs()

# mlp headline + linear control, both L2-normed
VARIANTS = [
    ("mlp", True, "mlp"),
    ("linear", True, "linear"),
]


def _evals_for_model(model) -> list:
    """Two probe variants (mlp headline + linear control) per model; no extract phase."""
    src_cache = cohort_paths(RUNS_OUT_CCB, model.slug, "ctrate").cache
    tgt_cache = cohort_paths(RUNS_OUT_CCB, model.slug, "radchestct").cache
    out = cohort_paths(RUNS_OUT, model.slug, "deploy")
    evals = []
    for slug, l2norm, head_kind in VARIANTS:
        evals.append(crossval_deploy_macro_eval(
            wrapper=model.wrapper, dataset="ctrate_deploy",
            cache_dir=src_cache, per_class_labels=_SRC_LABELS,
            cache_dir_target=tgt_cache, per_class_labels_target=_TGT_LABELS,
            output_dir=out.variant(slug),
            cohort="ct_rate", cohort_target="radchestct",
            l2_normalize=l2norm, head_kind=head_kind,
            checkpoint_path=model.checkpoint, model_arch=model.arch,
            slurm_time="04:00:00",
        ))
    return evals


EXPERIMENTS = [
    Experiment(name=model.slug, evaluations=_evals_for_model(model))
    for model in MODELS
]
