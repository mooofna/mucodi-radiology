# MuCoDi-radiology environment. Edit the storage roots + Slurm settings marked EDIT,
# then run:  source jobs/env.sh   (the orchestrator's emitted sbatch scripts source it too).

export REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# Compiler for Triton (optional).
if command -v gcc >/dev/null 2>&1; then export CC="$(command -v gcc)"; fi
if command -v g++ >/dev/null 2>&1; then export CXX="$(command -v g++)"; fi

# Storage roots -- EDIT THESE. PROJECT_ROOT: persistent (caches + weights). SCRATCH_ROOT: working (data + runs).
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT/.local/persistent}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-$REPO_ROOT/.local/scratch}"

export DATA_ROOT="$SCRATCH_ROOT/data"
export RUNS_ROOT="$SCRATCH_ROOT/runs"                 # experiments/runs symlinks here
export WANDB_DIR="$RUNS_ROOT"
export HF_HOME="$PROJECT_ROOT/cache/huggingface"      # model weights (persistent)
export TORCH_HOME="$PROJECT_ROOT/cache/torch"
export WANDB_CACHE_DIR="$PROJECT_ROOT/cache/wandb"
export UV_CACHE_DIR="$PROJECT_ROOT/cache/uv"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/pip"
mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$HF_HOME" "$TORCH_HOME" "$WANDB_CACHE_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

export RATE_SKIP_CACHE_META_CHECK=1

# Slurm defaults the orchestrator reads (experiments/cluster.py) -- EDIT partition + account.
export MUCODI_SLURM_PARTITION="${MUCODI_SLURM_PARTITION:-gpu}"
export MUCODI_SLURM_ACCOUNT="${MUCODI_SLURM_ACCOUNT:-your-account}"
export MUCODI_SLURM_QOS="${MUCODI_SLURM_QOS:-}"                       # blank = omit --qos
export MUCODI_SLURM_MAIL_USER="${MUCODI_SLURM_MAIL_USER:-}"           # blank = omit --mail-*
export MUCODI_TRAIN_CPUS="${MUCODI_TRAIN_CPUS:-32}"                   # CPUs per training node
export MUCODI_CPUS_PER_TASK="${MUCODI_CPUS_PER_TASK:-8}"              # CPUs per extract/evaluate job
# export MUCODI_MEM="180G"

# The packages are not pip-installed -> put the repo on the import path.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/rate-evals:${PYTHONPATH:-}"

# Activate the venv (build it first: cd "$REPO_ROOT" && uv sync --frozen).
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"

# Secrets (HF_TOKEN / ZENODO / KAGGLE / REDIVIS), if present -- template: jobs/secrets.env.example.
if [ -f "$REPO_ROOT/jobs/secrets.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/jobs/secrets.env"
  set +a
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi

# Inside a Slurm job: force HF offline (stage downloads on a login node first), cap intra-op
# threads (parallelism comes from workers), and disable core dumps.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  ulimit -c 0 2>/dev/null || true
fi
