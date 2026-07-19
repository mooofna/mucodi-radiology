"""KD smoke training-only fixture covering the launcher's training leg (not a production study)."""
from experiments.config import Experiment, TrainingConfig

STUDY_NAME = "kd_smoke"

EXPERIMENTS = [
    Experiment(
        name="kd_smoke_mobileone",
        training=TrainingConfig(
            arch="mobileone_mu1",
            dataset_profile="ctrate_kd",
            train_steps=200,
            warmup_steps=20,
            save_every=100,
            num_gpus=1,
        ),
        evaluations=[],
    ),
]
