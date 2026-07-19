#!/usr/bin/env bash
# Stage external teacher code (TANGERINE, CT-CLIP, Pillar-0) + weights into scripts/external/ and $DATA_ROOT.
# Idempotent. Reads $DATA_ROOT from env (set it via jobs/env.sh); activate the venv first so the
# pip-installed CT_CLIP / transformer_maskgit packages land in it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/.local/scratch/data}"
EXT="$REPO_ROOT/scripts/external"
TANGERINE_DIR="$EXT/3d-mae-medimaging"
CTCLIP_DIR="$EXT/CT-CLIP"
TANGERINE_CACHE="$DATA_ROOT/radiology/verification/.tangerine_cache"
CTCLIP_CACHE="$DATA_ROOT/radiology/verification/.ctclip_cache"

# Pinned commit SHAs.
TANGERINE_SHA="d8946bd8d1d75f18b5985413d29238904d68ee08"
CTCLIP_SHA="a2a155c601987820433c01db69b64d701d3d229d"

TANGERINE_URL="https://zenodo.org/records/18835750/files/tangerine.pth?download=1"
TANGERINE_MD5="d594de5ec0f59c55a1c20ec9f6ad3bef"

step() { printf '\n>>> %s\n' "$*"; }

step "[1/7] Clone TANGERINE -> $TANGERINE_DIR"
mkdir -p "$EXT"
if [[ ! -d "$TANGERINE_DIR/.git" ]]; then
    git clone https://github.com/niccolo246/3D-MAE-MedImaging "$TANGERINE_DIR"
else
    echo "    (already cloned)"
fi
git -C "$TANGERINE_DIR" fetch --quiet origin
git -C "$TANGERINE_DIR" checkout --quiet "$TANGERINE_SHA"
echo "    pinned to $TANGERINE_SHA"

step "[2/7] Clone CT-CLIP -> $CTCLIP_DIR"
if [[ ! -d "$CTCLIP_DIR/.git" ]]; then
    git clone https://github.com/ibrahimethemhamamci/CT-CLIP "$CTCLIP_DIR"
else
    echo "    (already cloned)"
fi
git -C "$CTCLIP_DIR" fetch --quiet origin
git -C "$CTCLIP_DIR" checkout --quiet "$CTCLIP_SHA"
echo "    pinned to $CTCLIP_SHA"

step "[3/7] pip install -e CT-CLIP packages (--no-deps to protect torch/numpy pins)"
PIP="${PIP:-uv pip}"
$PIP install --no-deps -e "$CTCLIP_DIR/transformer_maskgit"
$PIP install --no-deps -e "$CTCLIP_DIR/CT_CLIP"
# Runtime deps for the CT-CLIP / transformer_maskgit forward path.
$PIP install \
    "einops>=0.8" \
    "vector-quantize-pytorch>=1.14" \
    "beartype>=0.18" \
    "accelerate>=0.30" \
    "ema-pytorch>=0.5" \
    "ftfy>=6.0" \
    "regex>=2023.0" \
    "torchtyping>=0.1"

step "[4/7] Download TANGERINE checkpoint (1.4 GB) -> $TANGERINE_CACHE/tangerine.pth"
mkdir -p "$TANGERINE_CACHE"
TANGERINE_PTH="$TANGERINE_CACHE/tangerine.pth"
if [[ ! -f "$TANGERINE_PTH" ]]; then
    wget --continue --output-document="$TANGERINE_PTH" "$TANGERINE_URL"
else
    echo "    (already present)"
fi
echo "    verifying MD5 ..."
echo "$TANGERINE_MD5  $TANGERINE_PTH" | md5sum -c -

step "[5/7] Download CT-CLIP checkpoints (3 x 1.77 GB) via huggingface_hub"
mkdir -p "$CTCLIP_CACHE"
export CTCLIP_CACHE
python - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
repo_id = "ibrahimhamamci/CT-RATE"
files = ["CT-CLIP_v2.pt", "CT_VocabFine_v2.pt", "CT_LiPro_v2.pt"]
for fn in files:
    print(f"    fetching {fn}", flush=True)
    p = hf_hub_download(
        repo_id=repo_id,
        filename=f"models/CT-CLIP-Related/{fn}",
        repo_type="dataset",
        local_dir=os.environ["CTCLIP_CACHE"],
    )
    print(f"    -> {p}", flush=True)
PYEOF

step "[6/7] Stage + patch Pillar-0 (its HF trust_remote_code code omits self.post_init())"
# Pillar-0 is pulled from the Hub on first use; fetch it now so its custom modeling code is
# cached under $HF_HOME/modules, then apply the missing post_init() (idempotent).
if [[ -z "${HF_HOME:-}" ]]; then
    echo "    (skipped: HF_HOME unset -- source jobs/env.sh, then re-run to stage + patch Pillar-0)"
else
    python - <<'PYEOF'
from transformers import AutoModel
print("    fetching YalaLab/Pillar0-ChestCT (caches its trust_remote_code module)", flush=True)
AutoModel.from_pretrained("YalaLab/Pillar0-ChestCT", trust_remote_code=True)
PYEOF
    python "$REPO_ROOT/scripts/patch_pillar0_post_init.py" || echo "    (patch reported a skip -- check output above)"
fi

step "[7/7] Done."
cat <<EOF

The teacher code and weights are now staged under the cache roots.

If 'BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")' 401's,
visit https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized and accept
the gated-model terms, then 'huggingface-cli login'.
EOF
