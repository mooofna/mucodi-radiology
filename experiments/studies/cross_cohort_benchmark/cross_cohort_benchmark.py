"""Cross-cohort external-generalisation benchmark: teachers, student rungs, random-init floor; mlp + linear probes."""
import os
from pathlib import Path

from experiments.config import Experiment, cv_5fold_eval, cv_5fold_macro_eval
from experiments.studies._layout import cohort_paths
from experiments.studies._roster import Model, build_models

STUDY_NAME = "cross_cohort_benchmark"

REPO = Path(os.environ.get("REPO_ROOT") or Path(__file__).resolve().parents[3])
DATA_EVAL = REPO / "data" / "evaluation"
RUNS_OUT = REPO / "experiments" / "runs" / STUDY_NAME
KD_RUNS = REPO / "experiments" / "runs" / "kd_student_sweep"


MODELS = build_models(KD_RUNS, "CCB_MODELS")


# Binary single-cell cohorts: (cohort_slug, dataset_yaml, cohort_key, labels_json, qa_key, wall).
BINARY_COHORTS = [
    ("lidc", "lidc_chest_ct", "lidc", "lidc_malignancy_labels.json",
     "Does this scan contain a malignant nodule (median radiologist rating > 3)?", "08:00:00"),
    # LIDC panel: cluster-median labels; thresholds Causey 2018, Shen 2019/HSCNN, presence>=2/marked>=4.
    ("lidc_spiculated", "lidc_chest_ct", "lidc", "lidc_labels__spiculated.json",
     "Does this scan contain a spiculated nodule (median radiologist rating >= 2)?", "01:30:00"),
    ("lidc_spiculated_marked", "lidc_chest_ct", "lidc", "lidc_labels__spiculated_marked.json",
     "Does this scan contain a markedly spiculated nodule (median rating >= 4)?", "01:30:00"),
    ("lidc_lobulated", "lidc_chest_ct", "lidc", "lidc_labels__lobulated.json",
     "Does this scan contain a lobulated nodule (median rating >= 2)?", "01:30:00"),
    ("lidc_lobulated_marked", "lidc_chest_ct", "lidc", "lidc_labels__lobulated_marked.json",
     "Does this scan contain a markedly lobulated nodule (median rating >= 4)?", "01:30:00"),
    ("lidc_subsolid", "lidc_chest_ct", "lidc", "lidc_labels__subsolid.json",
     "Does this scan contain a subsolid (non-solid / part-solid) nodule (median texture rating <= 3)?", "01:30:00"),
    ("lidc_calcified", "lidc_chest_ct", "lidc", "lidc_labels__calcified.json",
     "Does this scan contain a calcified nodule (median calcification rating <= 5, i.e. not absent)?", "01:30:00"),
    ("lidc_subtle", "lidc_chest_ct", "lidc", "lidc_labels__subtle.json",
     "Does this scan contain a subtle nodule (median subtlety rating <= 3)?", "01:30:00"),
    ("lidc_poorly_marginated", "lidc_chest_ct", "lidc", "lidc_labels__poorly_marginated.json",
     "Does this scan contain a poorly-marginated nodule (median margin rating <= 3)?", "01:30:00"),
    ("lidc_non_spherical", "lidc_chest_ct", "lidc", "lidc_labels__non_spherical.json",
     "Does this scan contain a non-spherical nodule (median sphericity rating <= 3)?", "01:30:00"),
    ("stoic2021", "stoic2021", "stoic", "stoic2021_labels__covid.json",
     "Is COVID-19 present?", "1-00:00:00"),
    ("stoic2021_severe", "stoic2021", "stoic", "stoic2021_labels__severe.json",
     "Is the COVID-19 case severe?", "04:00:00"),
    ("stoic2021_severe_among_covid", "stoic2021", "stoic", "stoic2021_labels__severe_among_covid.json",
     "Is the COVID-19 case severe? (among RT-PCR+)", "04:00:00"),
    ("rspect", "rspect", "rspect", "rspect_labels.json",
     "Is pulmonary embolism present? (RSPECT/RSNA-PE 2020 study-level, 1-negative_exam_for_pe)", "3-00:00:00"),
    # RSPECT panel: RSNA-STR train.csv endpoints; ESC 2019/Meinel 2015 strain, Ata 2022 central, Ende-Verhaar 2017 chronic.
    ("rspect_rv_lv_strain", "rspect", "rspect", "rspect_labels__rv_lv_strain.json",
     "Does this CT show right-heart strain (RV/LV diameter ratio >= 1)? (RSPECT/RSNA-PE 2020 study-level)", "24:00:00"),
    ("rspect_central_pe", "rspect", "rspect", "rspect_labels__central_pe.json",
     "Is a central (main/saddle) pulmonary embolism present? (RSPECT/RSNA-PE 2020 study-level)", "24:00:00"),
    ("rspect_rv_lv_strain_among_pe", "rspect", "rspect", "rspect_labels__rv_lv_strain_among_pe.json",
     "Among PE-positive scans, is there right-heart strain (RV/LV ratio >= 1)? (RSPECT/RSNA-PE 2020 study-level)", "24:00:00"),
    ("rspect_chronic_pe", "rspect", "rspect", "rspect_labels__chronic_pe.json",
     "Are chronic thromboembolic features present (chronic or acute-and-chronic PE)? (RSPECT/RSNA-PE 2020 study-level)", "24:00:00"),
    ("mosmed", "mosmed", "mosmed", "mosmed_labels__severity_moderate_or_worse.json",
     "Is the COVID severity moderate or worse (CT-2..CT-4)?", "02:30:00"),
    # labels derived from MosMed CT-0..4 grade
    ("mosmed_covid_findings", "mosmed", "mosmed", "mosmed_labels__covid_findings.json",
     "Does this CT show COVID-19 CT findings?", "02:30:00"),
    ("mosmed_severe", "mosmed", "mosmed", "mosmed_labels__severe.json",
     "Does this CT show severe COVID-19 lung involvement (CT-3/CT-4)?", "02:30:00"),
    ("coca", "coca", "coca", "coca_labels.json",
     "Is coronary artery calcification present? (0=No, 1=Yes)", "01:00:00"),
    ("coca_significant", "coca", "coca", "coca_labels__significant.json",
     "Is the coronary calcium burden clinically significant (Agatston >= 100)?", "01:00:00"),
    ("coca_high", "coca", "coca", "coca_labels__high.json",
     "Is the coronary calcium burden high/severe (Agatston >= 400)?", "01:00:00"),
    ("coca_lad", "coca", "coca", "coca_labels__lad.json",
     "Is calcification present in the LAD (left anterior descending) artery? (0=No, 1=Yes)", "01:00:00"),
    # COCA GATED: ECG-gated Agatston from calcium_xml (>=130 HU); Detrano 2008, ACC/AHA 2018 (>=100), Rumberger 1999 (>=400), Hecht 2018.
    ("coca_gated", "coca_gated", "coca_gated", "coca_gated_labels.json",
     "Is coronary artery calcification present? (0=No, 1=Yes)", "01:00:00"),
    ("coca_gated_significant", "coca_gated", "coca_gated", "coca_gated_labels__significant.json",
     "Is the coronary calcium burden clinically significant (Agatston >= 100)?", "01:00:00"),
    ("coca_gated_high", "coca_gated", "coca_gated", "coca_gated_labels__high.json",
     "Is the coronary calcium burden high/severe (Agatston >= 400)?", "01:00:00"),
    ("coca_gated_lad", "coca_gated", "coca_gated", "coca_gated_labels__lad.json",
     "Is calcification present in the LAD (left anterior descending) artery? (0=No, 1=Yes)", "01:00:00"),
    ("coca_gated_lcx", "coca_gated", "coca_gated", "coca_gated_labels__lcx.json",
     "Is calcification present in the LCX (left circumflex) artery? (0=No, 1=Yes)", "01:00:00"),
    ("coca_gated_rca", "coca_gated", "coca_gated", "coca_gated_labels__rca.json",
     "Is calcification present in the RCA (right coronary) artery? (0=No, 1=Yes)", "01:00:00"),
    ("coca_gated_lca", "coca_gated", "coca_gated", "coca_gated_labels__lca.json",
     "Is calcification present in the LCA (left main / left coronary) artery? (0=No, 1=Yes)", "01:00:00"),
    ("coca_gated_multivessel", "coca_gated", "coca_gated", "coca_gated_labels__multivessel.json",
     "Is multivessel coronary calcification present (>=2 of LAD/LCX/RCA/LCA)? (0=No, 1=Yes)", "01:00:00"),
    ("osic", "osic", "osic", "osic_labels.json",
     "Did this IPF patient experience >=10% FVC decline within 52 weeks of baseline?", "01:00:00"),
    ("osic_ppf", "osic", "osic", "osic_labels__ppf.json",
     "Did FVC%predicted decline by >=5 absolute points within 52 weeks (ATS/ERS 2022 PPF physiological progression)?", "01:00:00"),
    ("osic_moderate", "osic", "osic", "osic_labels__moderate.json",
     "Is baseline FVC%predicted < 75% (GAP physiology-domain impairment)?", "01:00:00"),
    ("dlcs24", "dlcs24", "dlcs24", "dlcs24_labels.json",
     "Was lung cancer subsequently diagnosed (any timing)? (0=No, 1=Yes)", "16:00:00"),
    ("dlcs24_lungrads3", "dlcs24", "dlcs24", "dlcs24_labels__lungrads3.json",
     "Is this a Lung-RADS positive screen (category >=3)? (0=No, 1=Yes)", "06:00:00"),
    ("dlcs24_lungrads4", "dlcs24", "dlcs24", "dlcs24_labels__lungrads4.json",
     "Is this Lung-RADS 4 (4A/4B/4X, suspicious for malignancy)? (0=No, 1=Yes)", "06:00:00"),
    # LNDb (Pedrosa 2019, Zenodo 6613714): author-GT Fleischner + nodule panel, n=236 train; Fleischner 2017, subsolid mean<=11/3, large>=250mm3.
    ('lndb', "lndb", "lndb", 'lndb_labels.json',
     'Does this chest CT warrant nodule follow-up (Fleischner 2017 category 1/2/3 vs 0)? (0=No, 1=Yes)', '04:00:00'),
    ('lndb_fleischner_urgent', "lndb", "lndb", 'lndb_labels__fleischner_urgent.json',
     'Is this the highest Fleischner follow-up urgency (category 3: 3-month CT / PET-CT / biopsy)? (0=No, 1=Yes)', '02:00:00'),
    ('lndb_subsolid', "lndb", "lndb", 'lndb_labels__subsolid.json',
     'Does this chest CT contain a subsolid nodule (author non-solid/part-solid texture class, mean rating < 11/3)? (0=No, 1=Yes)', '02:00:00'),
    ('lndb_large_nodule', "lndb", "lndb", 'lndb_labels__large_nodule.json',
     'Does this chest CT contain a large nodule (author volume class, consolidated volume >= 250 mm3)? (0=No, 1=Yes)', '02:00:00'),
    ('lndb_nodule_present', "lndb", "lndb", 'lndb_labels__nodule_present.json',
     'Does this chest CT contain a pulmonary nodule (>= 1 consolidated true nodule)? (0=No, 1=Yes)', '02:00:00'),
    ('lndb_nodule_agreement', "lndb", "lndb", 'lndb_labels__nodule_agreement.json',
     'Does this chest CT contain a multi-reader-consensus nodule (>= 2 radiologists agreed)? (0=No, 1=Yes)', '02:00:00'),
    # MIDRC-RICORD (Tsai 2021, TCIA/IDC; Simpson 2020): 240 studies / 227 subjects (1A COVID+ / 1B COVID-), RT-PCR reference.
    ('midrc_ricord', "midrc_ricord", "midrc", 'midrc_ricord_labels.json',
     'Is COVID-19 present on this chest CT (RT-PCR reference standard; MIDRC-RICORD 1A vs 1B)? (0=No, 1=Yes)', '04:00:00'),
    ('midrc_ricord_typical', "midrc_ricord", "midrc", 'midrc_ricord_labels__typical.json',
     'Is this chest CT an RSNA-typical appearance for COVID-19 pneumonia (Typical vs Indeterminate/Atypical/Negative; higher-specificity cut)? (0=No, 1=Yes)', '02:00:00'),
]

# Diagnosis-macro anchors (in-corpus, NOT held-out): (slug, dataset, key, per_class_glob, wall).
MACRO_COHORTS = [
    ("ctrate", "ctrate", "ct_rate", "ctrate_labels__*.json", "06:00:00"),
    ("radchestct", "radchestct", "radchestct", "radchestct16_labels__*.json", "06:00:00"),
]

# Optional env cohort subset (smoke tests); matched against BOTH lists' slugs.
_COHORT_KEEP = {c.strip() for c in os.environ.get("CCB_COHORTS", "").split(",") if c.strip()} or None

# drop mosaic_attenuation_pattern (no Draelos token) -> canonical 16 for the macro aggregator
_MACRO_EXCLUDE = ("mosaic_attenuation_pattern",)


def _per_class_paths(glob: str) -> list[str]:
    _exclude = _MACRO_EXCLUDE if "radchestct" in glob else ()
    files = [p for p in sorted(DATA_EVAL.glob(glob))
             if not any(x in p.name for x in _exclude)]
    if not files:
        raise FileNotFoundError(
            f"no per-class label JSONs match {glob!r} under {DATA_EVAL} -- run the "
            f"dataprep label build for that cohort first."
        )
    return [str(p) for p in files]


# (variant_slug, l2_normalize, head_kind)
VARIANTS: list[tuple[str, bool, str]] = [
    ("mlp", True, "mlp"),
    ("linear", True, "linear"),
]


def _evals_for_model(model: Model) -> list:
    """One shared cache + (mlp, linear) output dirs per cohort, keyed on model.slug."""
    evals = []
    # cache keyed on dataset, output on cohort -> label-task cells share one extraction
    for cohort, dataset, key, labels_json, qa_key, wall in BINARY_COHORTS:
        if _COHORT_KEEP is not None and cohort not in _COHORT_KEEP:
            continue
        cache = cohort_paths(RUNS_OUT, model.slug, dataset).cache
        out = cohort_paths(RUNS_OUT, model.slug, cohort)
        for slug, l2norm, head_kind in VARIANTS:
            evals.append(cv_5fold_eval(
                wrapper=model.wrapper, dataset=dataset,
                cache_dir=cache, labels_json=DATA_EVAL / labels_json,
                output_dir=out.variant(slug), qa_key=qa_key, num_classes=2,
                cohort=key, l2_normalize=l2norm, l2_grid=False, head_kind=head_kind,
                checkpoint_path=model.checkpoint, model_arch=model.arch,
                slurm_time=wall,
            ))
    for cohort, dataset, key, glob, wall in MACRO_COHORTS:
        if _COHORT_KEEP is not None and cohort not in _COHORT_KEEP:
            continue
        cache = cohort_paths(RUNS_OUT, model.slug, dataset).cache
        out = cohort_paths(RUNS_OUT, model.slug, cohort)
        for slug, l2norm, head_kind in VARIANTS:
            evals.append(cv_5fold_macro_eval(
                wrapper=model.wrapper, dataset=dataset,
                cache_dir=cache, per_class_labels=_per_class_paths(glob),
                output_dir=out.variant(slug), cohort=key,
                l2_normalize=l2norm, l2_grid=False, head_kind=head_kind,
                checkpoint_path=model.checkpoint, model_arch=model.arch,
                slurm_time=wall,
            ))
    # 3D volumes can't be batch-collated -> B=1 on 1 GPU per cell
    for ec in evals:
        ec.extract_batch_size = 1
        ec.extract_num_gpus = 1
        ec.extract_num_workers = 12
        ec.cpus_per_task = 16
    return evals


EXPERIMENTS = [
    Experiment(name=model.slug, evaluations=_evals_for_model(model))
    for model in MODELS
]
