"""OmegaConf-based configuration management with Hydra support."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig, ListConfig, OmegaConf, MISSING
import os


def load_config(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    overrides: Optional[list] = None,
) -> DictConfig:
    """Load configuration using OmegaConf with Hydra-style `defaults:` composition."""
    project_root = Path(__file__).parent.parent.parent

    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    cfg = OmegaConf.load(config_path)

    def resolve_config_path(group_name: str, config_name: str):
        """Resolve config path supporting both singular and plural directory names."""
        wrap_group = None

        if group_name in ("model", "models"):
            search_groups = ["models", "model"]
            wrap_group = "model"
        elif group_name in ("dataset", "datasets"):
            search_groups = ["datasets", "dataset"]
            wrap_group = "dataset"
        else:
            search_groups = [group_name]

        for group_dir in search_groups:
            candidate_path = project_root / "configs" / group_dir / f"{config_name}.yaml"
            if candidate_path.exists():
                return candidate_path, wrap_group

        # fallback to a flat configs/<name>.yaml for legacy layouts
        direct_path = project_root / "configs" / f"{config_name}.yaml"
        if direct_path.exists():
            return direct_path, wrap_group

        return None, wrap_group

    if "defaults" in cfg:
        defaults = cfg.defaults
        composed_configs = []

        for default in defaults:
            if isinstance(default, str):
                if default == "_self_":
                    continue

                if "/" in default:
                    group, name = default.split("/", 1)

                    if group == "model" and model_name:
                        name = model_name
                    elif group == "dataset" and dataset_name:
                        name = dataset_name

                    config_file, wrap_group = resolve_config_path(group, name)
                else:
                    config_file = project_root / "configs" / f"{default}.yaml"
                    wrap_group = None

                if config_file and config_file.exists():
                    loaded_config = OmegaConf.load(config_file)
                    # wrap under its group so the @package directive resolves
                    if wrap_group in ["model", "dataset"]:
                        wrapped_config = OmegaConf.create({wrap_group: loaded_config})
                        composed_configs.append(wrapped_config)
                    else:
                        composed_configs.append(loaded_config)
                elif config_file is None:
                    raise FileNotFoundError(
                        f"Config file not found for group '{default}' (searched in singular/plural directories)"
                    )
                else:
                    raise FileNotFoundError(f"Config file not found: {config_file}")
            elif isinstance(default, dict) or isinstance(default, DictConfig):
                for group, name in default.items():
                    if group == "_self_":
                        continue

                    if group == "model" and model_name:
                        name = model_name
                    elif group == "dataset" and dataset_name:
                        name = dataset_name

                    config_file, wrap_group = resolve_config_path(group, name)

                    if config_file and config_file.exists():
                        loaded_config = OmegaConf.load(config_file)
                        # wrap under its group so the @package directive resolves
                        if wrap_group in ["model", "dataset"]:
                            wrapped_config = OmegaConf.create({wrap_group: loaded_config})
                            composed_configs.append(wrapped_config)
                        else:
                            composed_configs.append(loaded_config)
                    else:
                        raise FileNotFoundError(
                            f"Config file not found for group '{group}' and name '{name}'"
                        )

        if composed_configs:
            final_cfg = OmegaConf.merge(*composed_configs, cfg)
        else:
            final_cfg = cfg

        if "defaults" in final_cfg:
            del final_cfg.defaults
    else:
        final_cfg = cfg

    if overrides:
        for override in overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                try:
                    # parse the override value as YAML; fall back to the raw string below
                    parsed_value = OmegaConf.create({key: value})[key]
                    nested_dict = {}
                    keys = key.split(".")
                    current = nested_dict
                    for k in keys[:-1]:
                        current[k] = {}
                        current = current[k]
                    current[keys[-1]] = parsed_value
                    final_cfg = OmegaConf.merge(final_cfg, OmegaConf.create(nested_dict))
                except:
                    nested_dict = {}
                    keys = key.split(".")
                    current = nested_dict
                    for k in keys[:-1]:
                        current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
                    final_cfg = OmegaConf.merge(final_cfg, OmegaConf.create(nested_dict))

    return final_cfg


def load_dataset_config(dataset_name: str) -> DictConfig:
    """Load a dataset-specific config YAML by name."""
    project_root = Path(__file__).parent.parent.parent
    search_dirs = ["datasets", "dataset"]

    for directory in search_dirs:
        config_path = project_root / "configs" / directory / f"{dataset_name}.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)

    raise FileNotFoundError(
        f"Dataset config not found in directories {search_dirs} for dataset '{dataset_name}'"
    )


def load_model_config(model_name: str) -> DictConfig:
    """Load a model-specific config YAML by name."""
    project_root = Path(__file__).parent.parent.parent
    search_dirs = ["models", "model"]

    for directory in search_dirs:
        config_path = project_root / "configs" / directory / f"{model_name}.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)

    raise FileNotFoundError(
        f"Model config not found in directories {search_dirs} for model '{model_name}'"
    )


def get_config_value(config: DictConfig, key: str) -> Any:
    """Get a value by dot-notation key (e.g. 'data.root_dir'); raises ValueError if missing."""
    try:
        value = OmegaConf.select(
            config,
            key,
            default=MISSING,
            throw_on_missing=False,
        )
        if value is MISSING:
            raise ValueError(f"Key '{key}' not found in config")
        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
        return value
    except Exception as exc:
        raise ValueError(f"Key '{key}' not found in config") from exc


def merge_configs(*configs: Union[Dict[str, Any], DictConfig]) -> DictConfig:
    """Merge configs; later configs override earlier ones."""
    omega_configs = []
    for config in configs:
        if isinstance(config, dict):
            omega_configs.append(OmegaConf.create(config))
        else:
            omega_configs.append(config)

    return OmegaConf.merge(*omega_configs)


def setup_pipeline(config_path: Optional[str] = None, **overrides) -> DictConfig:
    """Set up the complete configuration pipeline, backfilling required sections with defaults."""
    model_name = overrides.pop("model", None)
    dataset_name = overrides.pop("dataset", None)

    override_list = []
    for key, value in overrides.items():
        override_list.append(f"{key}={value}")

    config = load_config(
        config_path=config_path,
        model_name=model_name,
        dataset_name=dataset_name,
        overrides=override_list if override_list else None,
    )

    if not hasattr(config, "hardware"):
        config.hardware = OmegaConf.create({"device": "cuda", "batch_size_per_gpu": 16})

    if not hasattr(config, "logging"):
        config.logging = OmegaConf.create({"level": "INFO"})

    if not hasattr(config, "model"):
        config.model = OmegaConf.create({"name": "unknown"})

    if not hasattr(config, "dataset"):
        config.dataset = OmegaConf.create({"name": "unknown"})

    return config
