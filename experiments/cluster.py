"""SLURM defaults, overridable via ``MUCODI_*`` env vars (set in jobs/env.sh)."""
from __future__ import annotations

import os

SLURM_PARTITION = os.environ.get("MUCODI_SLURM_PARTITION", "gpu")
SLURM_ACCOUNT = os.environ.get("MUCODI_SLURM_ACCOUNT", "")  # empty -> no --account emitted
SLURM_QOS = os.environ.get("MUCODI_SLURM_QOS", "")  # empty -> no --qos emitted
SLURM_CONSTRAINT = os.environ.get("MUCODI_SLURM_CONSTRAINT", "")  # empty -> not emitted
SLURM_EXCLUDE = os.environ.get("MUCODI_SLURM_EXCLUDE", "")  # empty -> not emitted
SLURM_MAIL_USER = os.environ.get("MUCODI_SLURM_MAIL_USER", "")  # empty -> no --mail-* emitted
DEFAULT_CPUS_PER_TASK = int(os.environ.get("MUCODI_CPUS_PER_TASK", "12"))
DEFAULT_TRAIN_CPUS = int(os.environ.get("MUCODI_TRAIN_CPUS", "256"))  # set to node core count
DEFAULT_MEM = os.environ.get("MUCODI_MEM", "180G")
