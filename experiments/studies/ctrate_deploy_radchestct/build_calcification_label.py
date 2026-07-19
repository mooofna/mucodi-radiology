"""Build the merged CT-RATE calcification source label as arterial OR coronary (Draelos-faithful)."""

from __future__ import annotations

import json
import os
from pathlib import Path

# parents[3] = repo root
REPO = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[3])
DATA_EVAL = REPO / "data" / "evaluation"
OUT_DIR = Path(__file__).resolve().parents[0] / "results" / "derived_labels"
OUT_PATH = OUT_DIR / "ctrate_labels__calcification.json"

_ARTERIAL = DATA_EVAL / "ctrate_labels__arterial_wall_calcification.json"
_CORONARY = DATA_EVAL / "ctrate_labels__coronary_artery_wall_calcification.json"
_QUESTION = "Is calcification present?"


def _answer(rec: dict) -> int:
    qa = (rec.get("qa_results") or {}).get("default_qa") or [{}]
    return int(list(qa[0].values())[0]) if qa and qa[0] else 0


def build() -> Path:
    arterial = json.loads(_ARTERIAL.read_text())
    coronary = json.loads(_CORONARY.read_text())
    accessions = sorted(set(arterial) & set(coronary))
    if not accessions:
        raise SystemExit(f"no shared accessions between {_ARTERIAL.name} and {_CORONARY.name}")

    merged: dict[str, dict] = {}
    n_pos = 0
    for acc in accessions:
        a, c = arterial[acc], coronary[acc]
        val = 1 if (_answer(a) == 1 or _answer(c) == 1) else 0
        n_pos += val
        merged[acc] = {
            "split": a.get("split", c.get("split")),
            "patient_id": a.get("patient_id", c.get("patient_id")),
            "qa_results": {"default_qa": [{_QUESTION: val}]},
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(merged, indent=2))
    print(f"wrote {OUT_PATH} -- n={len(merged)}, pos={n_pos}")
    return OUT_PATH


if __name__ == "__main__":
    build()
