"""Evidence data for the web demo, built only from thesis artifacts, with provenance.

Every value the page shows is traceable: each output carries a ``sources`` map (repository-relative
path, sha256, and the thesis table or figure it reproduces). Values recomputed from raw results are
checked against the thesis tables (tidy CSV exports) at display precision; a mismatch fails the
build. Aggregates only: no per-case data from any evaluation cohort is written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mucodi_demo.config import Paths

RUNG_LABELS = {
    "tmin": "TMIN",
    "t0": "T0",
    "t1": "T1",
    "t2": "T2",
    "t3": "T3",
    "t4": "T4",
    "t5": "T5",
    "t6": "T6",
    "b0": "B0",
    "b2": "B2",
    "b3": "B3",
    "b4": "B4",
    "b5": "B5",
    "b6": "B6",
}
LABEL_TO_RUNG = {v: k for k, v in RUNG_LABELS.items()}


class EvidenceMismatch(AssertionError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Sources:
    paths: Paths
    entries: dict[str, dict[str, str]] = field(default_factory=dict)

    def add(self, key: str, path: Path, label: str) -> Path:
        root = self.paths.thesis_code.parent
        try:
            rel = str(path.relative_to(root))
        except ValueError:
            rel = path.name
        self.entries[key] = {"file": rel, "sha256": _sha256(path), "label": label}
        return path


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _check(name: str, ours: float, thesis: float, digits: int) -> None:
    if round(ours, digits) != round(thesis, digits):
        raise EvidenceMismatch(
            f"{name}: recomputed {ours:.6f} rounds differently from thesis {thesis}"
        )


def build_inference(src: Sources) -> dict[str, Any]:
    """Thesis Table 6.5 from inference_raw.json, cross-checked against the table export."""
    raw_path = src.add(
        "inference_raw",
        src.paths.thesis_code
        / "experiments/studies/inference_benchmark/results/inference_raw.json",
        "Raw inference benchmark (thesis Table 6.5 and Figure 6.6 source)",
    )
    table_path = src.add(
        "table6_5", src.paths.thesis_tables / "inference_table6_5.csv", "Thesis Table 6.5"
    )
    teachers_path = src.add(
        "table3_1", src.paths.thesis_tables / "teachers_table3_1.csv", "Thesis Table 3.1"
    )
    sweep_path = src.add(
        "table4_1", src.paths.thesis_tables / "sweep_table4_1.csv", "Thesis Table 4.1"
    )
    raw = json.loads(raw_path.read_text())
    table = {r["model"]: r for r in _csv(table_path)}
    teachers = {r["model"]: r for r in _csv(teachers_path)}
    sweep = {r["model"]: r for r in _csv(sweep_path)}

    p0 = next(r for r in raw["records"] if r["key"] == "pillar0_chest_ct")
    records = []
    gpu_mismatches: list[dict[str, Any]] = []
    for r in raw["records"]:
        label = r["label"]
        cpu = r.get("cpu_latency_median")
        row = table.get("CT-CLIP (VocabFine)" if label == "CT-CLIP (VocabFine)" else label)
        entry: dict[str, Any] = {
            "key": r["key"],
            "label": label,
            "family": r["family"],
            "flops_g": r["flops_g"],
            "cpu_median_ms": cpu,
            "cpu_p90_ms": r.get("cpu_latency_p90"),
            "gpu_peak_throughput_raw": r.get("peak_throughput_vol_s"),
            "src": "inference_raw",
        }
        # Parameter counts as reported in the thesis (the raw record counts extra modules for
        # some teachers, e.g. Pillar-0's unused text tower: 94.1 M raw vs 79 M in Table 3.1).
        if label in sweep:
            entry["params_m"] = float(sweep[label]["params_M"])
            entry["params_src"] = "table4_1"
        elif label in teachers:
            entry["params_m"] = float(teachers[label]["params_M"])
            entry["params_src"] = "table3_1"
        else:
            entry["params_m"] = r["params_m"]
            entry["params_src"] = "inference_raw"
        entry["params_raw_m"] = r["params_m"]
        if row is not None and cpu is not None:
            _check(f"{label} CPU s", cpu / 1000, float(row["cpu_s"]), 2)
            speedup = p0["cpu_latency_median"] / cpu
            _check(f"{label} CPU speed-up", speedup, float(row["cpu_speedup_vs_pillar0"]), 1)
            entry["cpu_speedup_vs_pillar0"] = speedup
        if row is not None:
            # GPU columns come from the thesis table itself: the repository's raw JSON holds a
            # different GPU run (e.g. Curia-1 704 ms raw vs 271.5 ms in Table 6.5); the
            # differences are listed in the verification ledger.
            entry["gpu_b1_ms"] = float(row["gpu_b1_ms"])
            entry["gpu_b1_speedup_vs_pillar0"] = float(row["gpu_b1_speedup"])
            entry["gpu_b16_ms"] = float(row["gpu_b16_ms"])
            entry["gpu_b16_speedup_vs_pillar0"] = float(row["gpu_b16_speedup"])
            entry["gpu_mem_gb"] = float(row["gpu_mem_gb"])
            entry["flop_cut_vs_pillar0"] = float(row["flop_cut_vs_pillar0"])
            entry["gpu_src"] = "table6_5"
            if round(r["latency_median"], 1) != float(row["gpu_b1_ms"]):
                gpu_mismatches.append(
                    {
                        "model": label,
                        "raw_ms": round(r["latency_median"], 2),
                        "table_ms": float(row["gpu_b1_ms"]),
                    }
                )
        rung = LABEL_TO_RUNG.get(label)
        if rung:
            entry["rung"] = rung
        records.append(entry)
    return {
        "cpu_envelope": raw["cpu_envelope"],
        "gpu_envelope": raw["gpu_envelope"],
        "records": records,
        "gpu_raw_vs_table": gpu_mismatches,
    }


def build_ladder(src: Sources) -> list[dict[str, Any]]:
    """Student ladder (Table 4.1, head budget, Appendix compute table)."""
    head = {
        r["model"]: r
        for r in _csv(
            src.add(
                "head_budget",
                src.paths.thesis_tables / "head_budget.csv",
                "Thesis head-budget table (Section 4.4)",
            )
        )
    }
    compute = {
        r["model"]: r
        for r in _csv(
            src.add(
                "tableA_3", src.paths.thesis_tables / "compute_tableA_3.csv", "Thesis Table A.3"
            )
        )
    }
    out = []
    for rung, label in RUNG_LABELS.items():
        h, c = head[label], compute[label]
        out.append(
            {
                "rung": rung,
                "label": label,
                "line": "t" if rung.startswith("t") else "b",
                "params_m": float(h["encoder_M"]),
                "d_s": int(h["d_S"]),
                "width": float(h["width"]),
                "depth": float(h["depth"]),
                "blocks": int(h["blocks"]),
                "gflops": float(h["gflops"]),
                "train_gpu_hours": float(c["gpu_hours"]),
                "train_gpus": int(c["gpus"]),
                "src": "head_budget",
            }
        )
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def clean(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, str):
            # House style: no en or em dashes, ranges are written with "to".
            return v.replace(chr(0x2013), " to ").replace(chr(0x2014), ", ")
        return v

    path.write_text(json.dumps(clean(payload), indent=1) + "\n")


def build_evidence(paths: Paths, out_dir: Path) -> None:
    src = Sources(paths)
    write_json(out_dir / "inference.json", build_inference(src))
    write_json(out_dir / "ladder.json", build_ladder(src))
    write_json(out_dir / "sources.json", src.entries)
