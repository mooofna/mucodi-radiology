"""JobSpec -> rendered SBATCH script round-trip tests."""

from pathlib import Path

from experiments import pipeline
from experiments.pipeline import _REPO_CODE, JobSpec, render_job


def test_jobspec_basic_render(tmp_path: Path) -> None:
    spec = JobSpec(
        name="test-job",
        log_dir=tmp_path / "logs",
        body="echo hello",
        time="01:30:00",
    )
    out = render_job(spec)

    assert "#SBATCH --job-name=test-job" in out
    assert "#SBATCH --time=01:30:00" in out
    assert "#SBATCH --partition=gpu" in out
    assert "#SBATCH --gres=gpu:1" in out
    assert "#SBATCH --cpus-per-task=12" in out
    assert "#SBATCH --mem=180G" in out
    assert "#SBATCH --nodes=1" in out
    assert "#SBATCH --ntasks-per-node=1" in out

    assert f"source {_REPO_CODE}/jobs/env.sh" in out
    assert 'cd "$REPO_ROOT"' in out

    assert "echo hello" in out

    assert "%j_out.log" in out
    assert "%j_err.log" in out


def test_jobspec_cpu_only_omits_gres(tmp_path: Path) -> None:
    spec = JobSpec(
        name="cpu-job",
        log_dir=tmp_path,
        body="echo hi",
        num_gpus=0,
    )
    out = render_job(spec)
    assert "#SBATCH --gres=" not in out


def test_jobspec_frozen(tmp_path: Path) -> None:
    spec = JobSpec(name="x", log_dir=tmp_path, body="")
    try:
        spec.name = "y"  # type: ignore[misc]
    except Exception as exc:  # noqa: BLE001
        assert "frozen" in str(exc).lower() or "FrozenInstanceError" in type(exc).__name__
    else:
        raise AssertionError("JobSpec should be frozen (immutable)")


def test_render_mail_directives_when_configured(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "SLURM_MAIL_USER", "user@example.com")
    spec = JobSpec(name="mail-test", log_dir=tmp_path, body="")
    out = render_job(spec)
    assert "#SBATCH --mail-type=FAIL" in out
    assert "#SBATCH --mail-user=user@example.com" in out


def test_render_log_dir_paths(tmp_path: Path) -> None:
    log_dir = tmp_path / "nested" / "logs"
    spec = JobSpec(name="paths", log_dir=log_dir, body="")
    out = render_job(spec)
    assert f"#SBATCH --output={log_dir}/%j_out.log" in out
    assert f"#SBATCH --error={log_dir}/%j_err.log" in out
