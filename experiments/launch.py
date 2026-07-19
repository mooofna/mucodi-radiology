"""Study launcher: render + submit each study's phase jobs (train -> extract -> evaluate -> aggregate)."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

from experiments.config import Experiment
from experiments.phase_detector import PhaseDetector, RadiologyPhaseDetector
from experiments.pipeline import (
    ExperimentJobs,
    JobSpec,
    PROJECT_ROOT,
    RUNS_DIR,
    build_experiment_jobs,
    render_job,
)


def _load_study(study_path: Path) -> tuple[str, list[Experiment]]:
    """Import a study file as a module; return (study_name, experiments)."""
    study_path = study_path.resolve()
    if not study_path.is_file():
        raise FileNotFoundError(f"Study file not found: {study_path}")
    spec = importlib.util.spec_from_file_location(f"_study_{study_path.stem}", study_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load study from {study_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    experiments = getattr(mod, "EXPERIMENTS", None)
    if not isinstance(experiments, list) or not all(isinstance(e, Experiment) for e in experiments):
        raise ValueError(f"{study_path} must declare EXPERIMENTS: list[Experiment]")
    name = getattr(mod, "STUDY_NAME", study_path.stem)
    for e in experiments:
        if e.study is None:
            e.study = name
    return name, experiments


def _sbatch(script_path: Path, dependency: str | None = None, dry_run: bool = False) -> str:
    """Submit a rendered Slurm script. Returns the job ID (or 'DRYRUN-...' when dry)."""
    if dry_run:
        return f"DRYRUN-{script_path.stem}"
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(script_path))
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _squeue_active_names(user: str | None = None) -> set[str]:
    """Return the set of job-names currently in the user's squeue."""
    if user is None:
        user = os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%j"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {ln.strip() for ln in out.splitlines() if ln.strip()}


def _scancel_by_name_prefix(prefixes: Iterable[str], user: str | None = None) -> None:
    """Cancel any active jobs whose name starts with one of the given prefixes."""
    if user is None:
        user = os.environ.get("USER", "")
    try:
        out = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", "%i %j"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    targets: list[str] = []
    prefix_tuple = tuple(prefixes)
    for ln in out.splitlines():
        parts = ln.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        jid, jname = parts
        if jname.startswith(prefix_tuple):
            targets.append(jid)
    if targets:
        subprocess.run(["scancel", *targets], check=False)


@contextlib.contextmanager
def _study_lock(study_dir: Path) -> Iterator[None]:
    """Exclusive lock guarding the squeue-read + sbatch block against double-submission."""
    study_dir.mkdir(parents=True, exist_ok=True)
    lock_path = study_dir / ".continue.lock"
    with open(lock_path, "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit(
                f"error: another launcher invocation holds {lock_path}. "
                "Wait for it to finish, or remove the lock file if stale."
            )
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _write_script(spec: JobSpec, jobs_dir: Path, prefix: str) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    script_path = jobs_dir / f"{prefix}_{spec.name}.sh"
    script_path.write_text(render_job(spec))
    script_path.chmod(0o755)
    return script_path


def _submit_experiment(
    exp: Experiment,
    jobs: ExperimentJobs,
    *,
    detector: PhaseDetector,
    skip_done: bool,
    dry_run: bool,
) -> dict[str, str]:
    """Submit one experiment's jobs, respecting the dependency graph + already-done phases."""
    run_dir = exp.run_dir()
    jobs_dir = run_dir / "jobs"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    exp.save()

    submitted: dict[str, str] = {}
    train_dep: str | None = None
    active_names = _squeue_active_names()

    if jobs.train is not None:
        if skip_done and detector.train_done(exp):
            print(f"  [skip] train: {exp.checkpoint_file()} exists")
        elif jobs.train.name in active_names:
            print(f"  [skip] train: {jobs.train.name} already in queue")
        else:
            script = _write_script(jobs.train, jobs_dir, prefix="01")
            jid = _sbatch(script, dry_run=dry_run)
            submitted["train"] = jid
            # stagger multi-node launches
            if not dry_run and getattr(getattr(exp, "training", None), "num_nodes", 1) > 1:
                time.sleep(20)
            train_dep = f"afterok:{jid}"
            print(f"  [submit] train: {jid}  ({script.name})")

    eval_jids: list[str] = []
    # several cells share one extract JobSpec; run it once, reuse the jid
    submitted_extract_jids: dict[str, str] = {}
    for cell_key, eval_spec in jobs.evaluates.items():
        eval_cfg = next(e for e in exp.evaluations if e.output_dir == cell_key)
        cell_short = f"{eval_cfg.wrapper}_{eval_cfg.dataset}"

        extract_dep: str | None = train_dep
        extract_spec = jobs.extracts.get(cell_key)
        if extract_spec is not None:
            if extract_spec.name in submitted_extract_jids:
                extract_dep = f"afterok:{submitted_extract_jids[extract_spec.name]}"
            elif skip_done and detector.extract_done(exp, eval_cfg):
                print(f"  [skip] extract {cell_short}: cache_meta.yaml present, finished")
            elif extract_spec.name in active_names:
                print(f"  [skip] extract {cell_short}: already in queue")
            else:
                script = _write_script(extract_spec, jobs_dir, prefix=f"02_{cell_short}")
                jid = _sbatch(script, dependency=extract_dep, dry_run=dry_run)
                submitted[f"extract:{cell_key}"] = jid
                submitted_extract_jids[extract_spec.name] = jid
                extract_dep = f"afterok:{jid}"
                print(f"  [submit] extract {cell_short}: {jid}")

        if skip_done and detector.evaluate_done(exp, eval_cfg):
            print(f"  [skip] evaluate {cell_short}: summary.json valid")
            continue
        if eval_spec.name in active_names:
            print(f"  [skip] evaluate {cell_short}: already in queue")
            continue
        dep = extract_dep
        script = _write_script(eval_spec, jobs_dir, prefix=f"03_{cell_short}")
        jid = _sbatch(script, dependency=dep, dry_run=dry_run)
        submitted[f"evaluate:{cell_key}"] = jid
        eval_jids.append(jid)
        print(f"  [submit] evaluate {cell_short} ({eval_cfg.protocol}): {jid}")

    if jobs.aggregate is not None:
        if skip_done and detector.aggregate_done(exp):
            print(f"  [skip] aggregate: outputs present")
        elif jobs.aggregate.name in active_names:
            print(f"  [skip] aggregate: already in queue")
        else:
            agg_dep = f"afterok:{':'.join(eval_jids)}" if eval_jids else None
            script = _write_script(jobs.aggregate, jobs_dir, prefix="04")
            jid = _sbatch(script, dependency=agg_dep, dry_run=dry_run)
            submitted["aggregate"] = jid
            print(f"  [submit] aggregate: {jid}")

    return submitted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit a radiology experiment study to a Slurm cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for examples.",
    )
    parser.add_argument("study", type=Path, help="Path to a study .py declaring EXPERIMENTS")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Render scripts to disk but do not submit")
    mode.add_argument("--rerun", action="store_true",
                      help="Cancel matching jobs + wipe run_dirs + submit fresh")
    mode.add_argument("--continue", dest="cont", action="store_true",
                      help="Resume missing phases under flock")
    args = parser.parse_args(argv)

    study_name, experiments = _load_study(args.study)
    study_dir = RUNS_DIR / study_name
    print(f"=== Study: {study_name} ({len(experiments)} experiments) ===")

    detector = RadiologyPhaseDetector()

    if args.rerun:
        prefixes = [f"train_{e.name}" for e in experiments]
        prefixes += [f"extract_{e.name}_" for e in experiments]
        prefixes += [f"eval_{e.name}_" for e in experiments]
        prefixes += [f"aggregate_{e.name}" for e in experiments]
        _scancel_by_name_prefix(prefixes)
        for e in experiments:
            rd = e.run_dir()
            if rd.exists():
                print(f"  [wipe] {rd}")
                shutil.rmtree(rd)

    skip_done = bool(args.cont)
    if args.cont:
        with _study_lock(study_dir):
            return _submit_all(experiments, detector, skip_done=True, dry_run=False)
    else:
        return _submit_all(experiments, detector, skip_done=skip_done, dry_run=args.dry_run)


def _auto_resume(exp) -> None:  # type: ignore[no-untyped-def]
    """Auto-discover resume_from from the latest checkpoint of an incomplete training rung."""
    t = getattr(exp, "training", None)
    if t is None or t.resume_from:
        return
    ckpt_dir = exp.run_dir() / "outputs" / "checkpoints"
    if not ckpt_dir.is_dir():
        return
    final = ckpt_dir / f"step_{t.train_steps:07d}.pth.tar"
    if final.exists():
        return
    cks = sorted(ckpt_dir.glob("step_*.pth.tar"))
    if cks:
        t.resume_from = str(cks[-1])
        print(f"  [resume] {exp.name}: resuming from {cks[-1].name}")


def _submit_all(
    experiments: list[Experiment],
    detector: PhaseDetector,
    *,
    skip_done: bool,
    dry_run: bool,
) -> int:
    for exp in experiments:
        print(f"\n--- Experiment: {exp.name} ---")
        if skip_done:
            _auto_resume(exp)
        jobs = build_experiment_jobs(exp)
        _submit_experiment(exp, jobs, detector=detector, skip_done=skip_done, dry_run=dry_run)
    print("\n=== Done ===")
    print(f"Monitor: squeue -u $USER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
