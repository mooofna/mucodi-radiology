"""Single source of truth for the kd_student_sweep parameter ladder."""
from __future__ import annotations

# teacher feature dims d_t
TEACHER_DIMS = {"pillar0_chest_ct": 1152, "curia1": 768}

# rung -> encoder params (M); b1 skipped
PARAMS_M = {
    "efficientnet3d_tmin": 0.0364,
    "efficientnet3d_t0": 0.083,
    "efficientnet3d_t1": 0.145,
    "efficientnet3d_t2": 0.27,
    "efficientnet3d_t3": 0.51,
    "efficientnet3d_t4": 1.04,
    "efficientnet3d_t5": 1.84,
    "efficientnet3d_t6": 3.11,
    "efficientnet3d_b0": 4.69,
    "efficientnet3d_b2": 8.72,
    "efficientnet3d_b3": 12.06,
    "efficientnet3d_b4": 19.62,
    "efficientnet3d_b5": 31.05,
    "efficientnet3d_b6": 44.40,
}

ALL_RUNGS = list(PARAMS_M)
