"""Shared model roster for the cross-cohort probe studies: teachers + landed student rungs + random-init floor."""
import os
from dataclasses import dataclass
from pathlib import Path

from experiments.studies.kd_student_sweep.ladder import ALL_RUNGS

FINAL_STEP = 73700
TEACHERS = [
    "pillar0_chest_ct",
    "tangerine_vit",
    "curia1",
    "curia2",
    "ctclip_zero_shot",
    "ctclip_vocabfine_zero_shot",
]
FLOOR_SEEDS = (0, 1)
CEILING_FLOOR_ARCH = "efficientnet3d_b6"


@dataclass(frozen=True)
class Model:
    """One benchmark model; ``slug`` is the unique path key (never the wrapper)."""

    slug: str
    wrapper: str
    arch: str | None = None
    checkpoint: str | None = None


def _rung_id(arch: str) -> str:
    """efficientnet3d_t0 -> t0 (the short slug used in paths + tables)."""
    return arch.replace("efficientnet3d_", "")


def _landed_archs(kd_runs: Path) -> set[str]:
    """Rungs whose FINAL student checkpoint exists under kd_runs."""
    return {
        arch for arch in ALL_RUNGS
        if (kd_runs / arch / "outputs" / "checkpoints" / f"step_{FINAL_STEP:07d}.pth.tar").is_file()
    }


def _student_models(kd_runs: Path) -> list[Model]:
    """One Model per rung whose final checkpoint exists (re-checked at build time)."""
    out: list[Model] = []
    for arch in sorted(_landed_archs(kd_runs), key=ALL_RUNGS.index):
        ckpt = kd_runs / arch / "outputs" / "checkpoints" / f"step_{FINAL_STEP:07d}.pth.tar"
        out.append(Model(slug=_rung_id(arch), wrapper="mucodi_student", arch=arch, checkpoint=str(ckpt)))
    return out


def _floor_models() -> list[Model]:
    """Random-init floor x 2 seeds; only the b6 (conservative-ceiling) floor is emitted."""
    out: list[Model] = []
    for arch in ALL_RUNGS:
        if arch != CEILING_FLOOR_ARCH:
            continue
        rung = _rung_id(arch)
        for seed in FLOOR_SEEDS:
            wrapper = "random_features" if seed == 0 else f"random_features_s{seed}"
            out.append(Model(slug=f"floor_{rung}_s{seed}", wrapper=wrapper, arch=arch))
    return out


def build_models(kd_runs: Path, subset_env: str) -> list[Model]:
    """Teachers + landed student rungs + random-init floor; env `subset_env` (CSV of slugs) restricts it."""
    models = [Model(slug=w, wrapper=w) for w in TEACHERS] + _student_models(kd_runs) + _floor_models()
    sel = os.environ.get(subset_env)
    if sel:
        keep = {s.strip() for s in sel.split(",") if s.strip()}
        models = [m for m in models if m.slug in keep]
    return models
