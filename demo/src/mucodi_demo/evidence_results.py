"""Thesis results for the evidence explorer: trade-off curve, cohorts, endpoints, curves, deploy.

Recomputed values are checked against the thesis tables at display precision (one decimal of
AUROC in percent). ROC and PR curves are aggregated from the out-of-fold predictions on fixed
grids; no patient identifiers, labels or scores of individual cases are written.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from mucodi_demo.evidence import EvidenceMismatch, Sources

MODEL_LABELS: dict[str, str] = {
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
    "pillar0_chest_ct": "Pillar-0",
    "curia1": "Curia-1",
    "curia2": "Curia-2",
    "ctclip_zero_shot": "CT-CLIP",
    "ctclip_vocabfine_zero_shot": "CT-CLIP-VF",
    "tangerine_vit": "TANGERINE",
}
FLOOR_LABEL = "Untrained (B6)"
FLOOR_SEEDS = ("floor_b6_s0", "floor_b6_s1")
# experiments/studies/kd_student_sweep/analysis/param_performance_curve.py::CC_EXTERNAL_UNITS
EXTERNAL_UNITS: list[tuple[str, list[str]]] = [
    ("LIDC", ["lidc_chest_ct"]),
    ("MosMed", ["mosmed"]),
    ("COCA", ["coca", "coca_gated"]),
    ("OSIC", ["osic"]),
    ("DLCS24", ["dlcs24"]),
    ("LNDb", ["lndb"]),
    ("STOIC", ["stoic2021"]),
    ("RSPECT", ["rspect"]),
    ("RAD_ChestCT", ["radchestct"]),
    ("MIDRC_RICORD", ["midrc_ricord"]),
]
T975_DF9 = 2.2621571627409915
# Cohort keys of cross_cohort_cells.csv -> columns of the thesis cohort tables.
COHORT_COLUMNS = {
    "lidc_chest_ct": "LIDC",
    "stoic2021": "STOIC",
    "rspect": "RSPECT",
    "mosmed": "MosMed",
    "coca": "COCA_nongated",
    "coca_gated": "COCA_gated",
    "osic": "OSIC",
    "dlcs24": "DLCS24",
    "lndb": "LNDb",
    "midrc_ricord": "MIDRC_RICORD",
    "radchestct": "RAD_ChestCT",
    "ctrate": "CT_RATE",
}
# Endpoint (cohort, name) in the thesis endpoint table -> Dropbox run folder (study slug).
ENDPOINT_DIRS: dict[tuple[str, str], str] = {
    ("lidc", "malig"): "lidc",
    ("lidc", "spic"): "lidc_spiculated",
    ("lidc", "spic+"): "lidc_spiculated_marked",
    ("lidc", "lob"): "lidc_lobulated",
    ("lidc", "lob+"): "lidc_lobulated_marked",
    ("lidc", "subsolid"): "lidc_subsolid",
    ("lidc", "calc"): "lidc_calcified",
    ("lidc", "subtle"): "lidc_subtle",
    ("lidc", "margin"): "lidc_poorly_marginated",
    ("lidc", "sphere"): "lidc_non_spherical",
    ("stoic2021", "COVID"): "stoic2021",
    ("stoic2021", "severe"): "stoic2021_severe",
    ("stoic2021", "sev|C+"): "stoic2021_severe_among_covid",
    ("rspect", "PE"): "rspect",
    ("rspect", "RV/LV"): "rspect_rv_lv_strain",
    ("rspect", "central"): "rspect_central_pe",
    ("rspect", "RV/LV|PE"): "rspect_rv_lv_strain_among_pe",
    ("rspect", "chronic"): "rspect_chronic_pe",
    ("mosmed", "mod+"): "mosmed",
    ("mosmed", "COVID"): "mosmed_covid_findings",
    ("mosmed", "severe"): "mosmed_severe",
    ("coca_nongated", "present"): "coca",
    ("coca_nongated", "geq100"): "coca_significant",
    ("coca_nongated", "geq400"): "coca_high",
    ("coca_nongated", "LAD"): "coca_lad",
    ("coca_gated", "present"): "coca_gated",
    ("coca_gated", "geq100"): "coca_gated_significant",
    ("coca_gated", "geq400"): "coca_gated_high",
    ("coca_gated", "LAD"): "coca_gated_lad",
    ("coca_gated", "LCX"): "coca_gated_lcx",
    ("coca_gated", "RCA"): "coca_gated_rca",
    ("coca_gated", "LCA"): "coca_gated_lca",
    ("coca_gated", "multi"): "coca_gated_multivessel",
    ("osic", "decline"): "osic",
    ("osic", "PPF"): "osic_ppf",
    ("osic", "FVC<75"): "osic_moderate",
    ("dlcs24", "cancer"): "dlcs24",
    ("dlcs24", "LRgeq3"): "dlcs24_lungrads3",
    ("dlcs24", "LR4"): "dlcs24_lungrads4",
    ("lndb", "actionable"): "lndb",
    ("lndb", "urgent"): "lndb_fleischner_urgent",
    ("lndb", "subsolid"): "lndb_subsolid",
    ("lndb", "large"): "lndb_large_nodule",
    ("lndb", "present"): "lndb_nodule_present",
    ("lndb", "consensus"): "lndb_nodule_agreement",
    ("midrc_ricord", "COVID"): "midrc_ricord",
    ("midrc_ricord", "typical"): "midrc_ricord_typical",
}
PER_CLASS_COHORTS = {"radchestct": "radchestct", "ctrate": "ctrate"}
CURVE_MODELS = ("tmin", "t2", "t6", "b0", "b3", "b6", "pillar0_chest_ct", "curia1")
GRID = np.linspace(0.0, 1.0, 201)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _cells(src: Sources) -> dict[tuple[str, str, str], dict[str, float]]:
    path = src.add(
        "cross_cohort_cells",
        src.paths.thesis_code
        / "experiments/studies/cross_cohort_benchmark/results/cross_cohort_cells.csv",
        "Cross-cohort cell results (thesis Tables 6.2 and 6.3 source)",
    )
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in _csv(path):
        out[(r["model"], r["cohort"], r["variant"])] = {
            k: float(r[k]) * 100 for k in ("auroc", "ci_lo", "ci_hi", "auprc", "f1")
        }
    return out


def _cell(
    cells: dict[tuple[str, str, str], dict[str, float]], model: str, cohort: str, probe: str
) -> dict[str, float]:
    if model == "floor":
        # Untrained floor: per cohort the higher of the two random-init seeds (thesis convention).
        cands = [cells[(s, cohort, probe)] for s in FLOOR_SEEDS if (s, cohort, probe) in cells]
        return max(cands, key=lambda c: c["auroc"])
    return cells[(model, cohort, probe)]


def build_tradeoff(src: Sources) -> dict[str, Any]:
    cells = _cells(src)
    tables = {
        "mlp": {
            r["model"]: r
            for r in _csv(
                src.add(
                    "table6_3",
                    src.paths.thesis_tables / "cohort_auroc_mlp_table6_3.csv",
                    "Thesis Table 6.3 (MLP probe)",
                )
            )
        },
        "linear": {
            r["model"]: r
            for r in _csv(
                src.add(
                    "table6_2",
                    src.paths.thesis_tables / "cohort_auroc_linear_table6_2.csv",
                    "Thesis Table 6.2 (linear probe)",
                )
            )
        },
    }
    models = [*MODEL_LABELS.keys(), "floor"]
    out: dict[str, Any] = {"units": [u for u, _ in EXTERNAL_UNITS], "probes": {}}
    for probe in ("mlp", "linear"):
        rows = []
        for m in models:
            label = FLOOR_LABEL if m == "floor" else MODEL_LABELS[m]
            units = {}
            for unit, cohorts in EXTERNAL_UNITS:
                units[unit] = float(np.mean([_cell(cells, m, c, probe)["auroc"] for c in cohorts]))
            vals = np.array(list(units.values()))
            mean = float(vals.mean())
            half = float(T975_DF9 * vals.std(ddof=1) / math.sqrt(len(vals)))
            row = tables[probe].get(label) or tables[probe].get(
                "Untrained" if m == "floor" else label
            )
            if row is not None:
                if round(mean, 1) != float(row["Avg_ext"]):
                    raise EvidenceMismatch(
                        f"{label} {probe} Avg_ext {mean:.3f} vs table {row['Avg_ext']}"
                    )
                if round(half, 1) != float(row["Avg_ext_ci"]):
                    raise EvidenceMismatch(
                        f"{label} {probe} Avg_ext spread {half:.3f} vs table {row['Avg_ext_ci']}"
                    )
                for cohort, col in COHORT_COLUMNS.items():
                    ours = _cell(cells, m, cohort, probe)["auroc"]
                    if col in row and row[col] != "" and round(ours, 1) != float(row[col]):
                        raise EvidenceMismatch(
                            f"{label} {probe} {cohort} {ours:.3f} vs table {row[col]}"
                        )
            rows.append(
                {
                    "model": m,
                    "label": label,
                    "avg_ext": mean,
                    "spread": half,
                    "units": units,
                    "checked": row is not None,
                }
            )
        out["probes"][probe] = rows
    out["spread_definition"] = "2.262 x SD / sqrt(10) over the ten external units"
    out["src"] = ["cross_cohort_cells", "table6_2", "table6_3"]
    return out


def build_cohorts(src: Sources) -> dict[str, Any]:
    cells = _cells(src)
    meta = _csv(
        src.add("table5_1", src.paths.thesis_tables / "cohorts_table5_1.csv", "Thesis Table 5.1")
    )
    rows = []
    for m in [*MODEL_LABELS.keys(), "floor"]:
        for cohort in COHORT_COLUMNS:
            for probe in ("mlp", "linear"):
                c = _cell(cells, m, cohort, probe)
                rows.append(
                    {
                        "model": FLOOR_LABEL if m == "floor" else MODEL_LABELS[m],
                        "cohort": cohort,
                        "probe": probe,
                        "auroc": round(c["auroc"], 2),
                        "ci": [round(c["ci_lo"], 2), round(c["ci_hi"], 2)],
                    }
                )
    return {
        "cohorts": meta,
        "columns": COHORT_COLUMNS,
        "cells": rows,
        "src": ["cross_cohort_cells", "table5_1"],
    }


_TEX = (
    (re.compile(r"\\SI\{([^}]*)\}\{\\percent\}"), r"\1 %"),
    (re.compile(r"\\geq\s+"), "\u2265 "),
    (re.compile(r"\\geq"), "\u2265"),
    (re.compile(r"\\leq\s+"), "\u2264 "),
    (re.compile(r"\\leq"), "\u2264"),
    (re.compile(r"\\vert\b\s*"), "|"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\[,\s]"), " "),
    (re.compile(r"\$"), ""),
)


def detex(text: str) -> str:
    """Plain text from the inline LaTeX of the thesis tables.

    Examples: ``$\\geq 3$``, ``\\SI{75}{\\percent}``, ``52\\,wk``.
    """
    for pattern, repl in _TEX:
        text = pattern.sub(repl, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def build_endpoints(src: Sources) -> dict[str, Any]:
    chars = _csv(
        src.add(
            "endpoint_characteristics",
            src.paths.thesis_tables / "endpoint_characteristics.csv",
            "Thesis endpoint characteristics (Chapter 5 and Appendix)",
        )
    )
    aucs = _csv(
        src.add(
            "endpoint_auroc",
            src.paths.thesis_tables / "endpoint_auroc.csv",
            "Thesis per-endpoint AUROC (Appendix profile tables)",
        )
    )
    out: dict[str, dict[str, Any]] = {}
    for c in chars:
        text = {
            k: detex(v) for k, v in c.items() if k in ("title", "name", "definition", "threshold")
        }
        out[f"{c['cohort']}|{c['endpoint']}"] = {**c, **text, "auroc": {}}
    for a in aucs:
        key = f"{a['cohort']}|{a['endpoint']}"
        if key not in out:
            out[key] = {"cohort": a["cohort"], "endpoint": a["endpoint"], "auroc": {}}
        out[key]["auroc"].setdefault(a["model"], {})[a["probe"]] = [
            float(a["auroc"]),
            float(a["half_width"]),
        ]
    return {"endpoints": list(out.values()), "src": ["endpoint_characteristics", "endpoint_auroc"]}


def _roc_pr(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    pos = int(y.sum())
    neg = len(y) - pos
    # Distinct thresholds: cumulative counts at the last index of each tied score block.
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = np.cumsum(1 - y)[distinct]
    tpr = np.r_[0.0, tp / pos]
    fpr = np.r_[0.0, fp / neg]
    auroc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    precision = np.r_[1.0, tp / np.maximum(tp + fp, 1)]
    recall = tpr
    # ROC on a fixed FPR grid (step interpolation, upper envelope) and PR with interpolated
    # precision.
    roc_grid = np.array([tpr[fpr <= g].max() for g in GRID])
    prec_env = np.maximum.accumulate(precision[::-1])[::-1]
    pr_grid = np.array([prec_env[recall >= g].max() if (recall >= g).any() else 0.0 for g in GRID])
    spec = 1 - fpr
    ops = []
    for name, idx in (
        ("sens90", int(np.argmax(np.where(tpr >= 0.9, spec, -1)))),
        ("spec90", int(np.argmax(np.where(spec >= 0.9, tpr, -1)))),
        ("youden", int(np.argmax(tpr - fpr))),
    ):
        ops.append(
            {"name": name, "sens": round(float(tpr[idx]), 4), "spec": round(float(spec[idx]), 4)}
        )
    return {
        "auroc": auroc,
        "n_pos": pos,
        "n_neg": neg,
        "roc_tpr": [round(float(v), 4) for v in roc_grid],
        "pr_precision": [round(float(v), 4) for v in pr_grid],
        "ops": ops,
    }


def _oof_path(runs: Path, model: str, cohort: str, endpoint: str, probe: str) -> Path | None:
    if cohort in PER_CLASS_COHORTS:
        return (
            runs
            / model
            / PER_CLASS_COHORTS[cohort]
            / probe
            / "per_class"
            / endpoint.replace(" ", "_")
            / "predictions_oof.json"
        )
    slug = ENDPOINT_DIRS.get((cohort, endpoint))
    return runs / model / slug / probe / "predictions_oof.json" if slug else None


def build_curves(src: Sources, out_dir: Path) -> dict[str, Any]:
    """Per-cohort curve files; every recomputed AUROC is checked against the thesis table."""
    runs = src.paths.dropbox_runs / "cross_cohort_benchmark"
    table: dict[tuple[str, str, str, str], float] = {}
    for a in _csv(src.paths.thesis_tables / "endpoint_auroc.csv"):
        table[(a["cohort"], a["endpoint"], a["model"], a["probe"])] = float(a["auroc"])
    endpoints = sorted({(c, e) for c, e, _, _ in table})
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"checked": 0, "mismatches": [], "missing": []}
    by_cohort: dict[str, dict[str, Any]] = {}
    for cohort, endpoint in endpoints:
        entry: dict[str, Any] = {"models": {}}
        for model in CURVE_MODELS:
            label = MODEL_LABELS[model]
            for probe in ("mlp", "linear"):
                path = _oof_path(runs, model, cohort, endpoint, probe)
                if path is None or not path.exists():
                    report["missing"].append(f"{cohort}|{endpoint}|{model}|{probe}")
                    continue
                rows = json.loads(path.read_text())
                labels = np.array([int(r["label"]) for r in rows])
                scores = np.array([float(r["score"]) for r in rows])
                curve = _roc_pr(labels, scores)
                thesis = table.get((cohort, endpoint, label, probe))
                if thesis is not None:
                    report["checked"] += 1
                    if round(curve["auroc"] * 100, 1) != thesis:
                        report["mismatches"].append(
                            f"{cohort}|{endpoint}|{label}|{probe}: "
                            f"{curve['auroc'] * 100:.2f} vs {thesis}"
                        )
                entry["n_pos"], entry["n_neg"] = curve.pop("n_pos"), curve.pop("n_neg")
                entry["models"].setdefault(label, {})[probe] = curve
        by_cohort.setdefault(cohort, {})[endpoint] = entry
    for cohort, eps in by_cohort.items():
        (out_dir / f"{cohort}.json").write_text(
            json.dumps({"cohort": cohort, "grid": 201, "endpoints": eps}) + "\n"
        )
    if report["mismatches"]:
        raise EvidenceMismatch(
            "curve AUROC does not reproduce the thesis: " + "; ".join(report["mismatches"][:10])
        )
    return report


def build_deploy(src: Sources) -> dict[str, Any]:
    table = _csv(
        src.add(
            "table6_4",
            src.paths.thesis_tables / "cold_deploy_table6_4.csv",
            "Thesis Table 6.4 (cold deploy)",
        )
    )
    raw = _csv(
        src.add(
            "deploy_cells",
            src.paths.thesis_code
            / "experiments/studies/ctrate_deploy_radchestct/results/deploy_cells.csv",
            "Cold-deploy cell results (Table 6.4 source)",
        )
    )
    raw_by = {(r["model"], r["variant"], r["side"]): float(r["macro_auroc"]) * 100 for r in raw}
    rows = []
    for t in table:
        rows.append(
            {
                "model": t["model"],
                "mlp": {
                    "internal": float(t["mlp_int_auroc"]),
                    "external": float(t["mlp_ext_auroc"]),
                    "external_ci": float(t["mlp_ext_auroc_ci"]),
                },
                "linear": {
                    "internal": float(t["lin_int_auroc"]),
                    "external": float(t["lin_ext_auroc"]),
                    "external_ci": float(t["lin_ext_auroc_ci"]),
                },
            }
        )
    # Spot check the raw cells against the table for the models present in both.
    for key, label in MODEL_LABELS.items():
        row = next((r for r in rows if r["model"] == label), None)
        raw = raw_by.get((key, "mlp", "external"))
        if row and raw is not None and round(raw, 1) != row["mlp"]["external"]:
            raise EvidenceMismatch(
                f"cold deploy {label}: raw {raw:.2f} vs table {row['mlp']['external']}"
            )
    return {"rows": rows, "src": ["table6_4", "deploy_cells"]}
