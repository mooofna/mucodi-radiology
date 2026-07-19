"""Configuration loader (OmegaConf + Hydra) for rate_eval."""

from .loader import (
    get_config_value,
    load_config,
    load_dataset_config,
    load_model_config,
    merge_configs,
    setup_pipeline,
)
from .hydra import (
    create_hydra_compatible_cli,
    get_hydra_overrides_from_env,
    load_config_with_hydra,
    parse_hydra_overrides_from_args,
    setup_hydra_config_dir,
)
from .dataset_schema import (
    DatasetConfig,
    LoaderSpec,
    PreprocessSpec,
)
from .overrides import apply_cli_overrides

__all__ = [
    "DatasetConfig",
    "LoaderSpec",
    "PreprocessSpec",
    "apply_cli_overrides",
    "create_hydra_compatible_cli",
    "get_config_value",
    "get_hydra_overrides_from_env",
    "load_config",
    "load_config_with_hydra",
    "load_dataset_config",
    "load_model_config",
    "merge_configs",
    "parse_hydra_overrides_from_args",
    "setup_hydra_config_dir",
    "setup_pipeline",
]
