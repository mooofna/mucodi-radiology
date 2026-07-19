"""Component factory for creating models and datasets."""

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf

from .core.errors import DatasetError, ModelError
from .core.logging import get_logger
from .config import load_dataset_config, load_model_config, merge_configs

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "tangerine_vit": ("rate_eval.models.tangerine_vit", "TangerineVit"),
    "ctclip_zero_shot": ("rate_eval.models.ctclip_zero_shot", "CTClipZeroShot"),
    "ctclip_vocabfine_zero_shot": ("rate_eval.models.ctclip_zero_shot", "CTClipZeroShot"),
    "curia2": ("rate_eval.models.curia2", "Curia2"),
    # Curia-1 (ViT-B): dedicated wrapper, not shared with curia2
    "curia1": ("rate_eval.models.curia1", "Curia1"),
    "pillar0_chest_ct": ("rate_eval.models.pillar0", "Pillar0"),
    "mucodi_student": ("rate_eval.models.student", "MuCoDiStudent"),
    # random-init floor: two names bake two init seeds (worst-of-2)
    "random_features": ("rate_eval.models.random_features", "RandomFeatures3D"),
    "random_features_s1": ("rate_eval.models.random_features", "RandomFeatures3D"),
}


def _to_dict(config_section: Any) -> Dict[str, Any]:
    """Convert OmegaConf sections to regular dictionaries."""
    if isinstance(config_section, DictConfig):
        return OmegaConf.to_container(config_section, resolve=True)  # type: ignore[return-value]
    if config_section is None:
        return {}
    if isinstance(config_section, dict):
        return config_section
    return {}


def _resolve_path(path_str: Union[str, Path]) -> Path:
    """Resolve a potentially relative path against the project root."""
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _load_dataset_class(name: str, loader_spec: Dict[str, Any] = None) -> type:
    """Load a dataset class from the datasets module (lazy-imported to stay torch-free)."""
    loader_spec = loader_spec or {}
    class_name = loader_spec.get("class")

    datasets_mod = import_module("rate_eval.datasets")

    if class_name and hasattr(datasets_mod, class_name):
        return getattr(datasets_mod, class_name)

    available_classes = [
        attr
        for attr in datasets_mod.__dict__
        if isinstance(datasets_mod.__dict__[attr], type)
        and hasattr(datasets_mod.__dict__[attr], "__module__")
        and datasets_mod.__dict__[attr].__module__.startswith("rate_eval.datasets")
    ]

    raise DatasetError(
        f"Dataset class for '{name}' not found. Available classes: {available_classes}"
    )


def _resolve_model_class(name: str) -> type:
    """Resolve the concrete model class from the registry."""
    try:
        module_path, class_name = _MODEL_REGISTRY[name]
    except KeyError as exc:
        available_models = sorted(_MODEL_REGISTRY)
        raise ModelError(f"Unknown model '{name}'. Available models: {available_models}") from exc

    try:
        module = import_module(module_path)
        model_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelError(
            f"Failed to import model '{name}' from '{module_path}.{class_name}': {exc}"
        ) from exc

    return model_cls


def _get_loader_spec(
    name: str, config: dict
) -> Tuple[Dict[str, Any], Optional[DictConfig], Dict[str, Any]]:
    """Return dataset loader spec along with any pre-loaded dataset config."""
    dataset_registry = _to_dict(getattr(config, "datasets", None))
    loader_spec = _to_dict(dataset_registry.get(name))
    if loader_spec:
        return loader_spec, None, dataset_registry

    try:
        dataset_config = load_dataset_config(name)
    except FileNotFoundError:
        return {}, None, dataset_registry

    loader_spec = _to_dict(OmegaConf.select(dataset_config, "loader"))
    if not loader_spec:
        # data-only spec (no loader:): synthesize from wrapper default_loader_class + preprocess
        model_dict = _to_dict(getattr(config, "model", None))
        default_cls = model_dict.get("default_loader_class")
        if default_cls:
            loader_spec = {"class": default_cls}
            wrapper_pp = model_dict.get("preprocess")
            if wrapper_pp:
                loader_spec["preprocess"] = wrapper_pp
            # source/npz_key live in the data-spec, not the wrapper
            for _k in ("source", "npz_key"):
                _v = OmegaConf.select(dataset_config, _k)
                if _v is not None:
                    if _k == "npz_key":
                        loader_spec.setdefault("preprocess", {})
                        loader_spec["preprocess"]["npz_key"] = _v
                    else:
                        loader_spec[_k] = _v
            OmegaConf.update(dataset_config, "loader", loader_spec, force_add=True)
    return loader_spec, dataset_config, dataset_registry


def _load_dataset_config_for_loader(
    name: str,
    loader_spec: Dict[str, Any],
    fallback_config: Optional[DictConfig],
) -> DictConfig:
    """Load the dataset config indicated by the loader spec."""
    dataset_config_path = loader_spec.get("config") or loader_spec.get("config_file")
    if dataset_config_path:
        resolved_path = _resolve_path(dataset_config_path)
        if not resolved_path.exists():
            raise DatasetError(
                f"Config file '{resolved_path}' for dataset '{name}' does not exist."
            )
        return OmegaConf.load(resolved_path)

    if fallback_config is not None:
        return fallback_config

    return load_dataset_config(name)


def create_model(name: str, config: dict) -> Any:
    """Create a model instance by registry name."""
    try:
        model_config = load_model_config(name)
        merged_config = merge_configs(model_config, config)

        model_cls = _resolve_model_class(name)

        logger.info("Creating %s model", model_cls.__name__)
        return model_cls(merged_config)

    except Exception as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelError(f"Failed to create model '{name}': {exc}") from exc


def create_dataset(
    name: str, config: dict, split: str = "train", transforms=None, model_preprocess=None
) -> Any:
    """Create a dataset instance by name (e.g. 'radchestct/ctclip_zero_shot')."""
    try:
        loader_spec, dataset_config_hint, dataset_registry = _get_loader_spec(name, config)

        if not loader_spec:
            available = sorted(dataset_registry.keys()) if dataset_registry else []
            raise DatasetError(
                f"Unknown dataset '{name}'. Provide loader configuration in YAML."
                f" Available datasets: {available}"
            )

        dataset_config = _load_dataset_config_for_loader(
            name,
            loader_spec,
            dataset_config_hint,
        )

        merged_config = merge_configs(dataset_config, config)

        dataset_class = _load_dataset_class(name, loader_spec)

        init_kwargs = loader_spec.get("init_args", {})
        if init_kwargs and not isinstance(init_kwargs, dict):
            raise DatasetError(f"init_args for dataset '{name}' must be a dictionary if provided.")

        logger.info(
            "Creating dataset '%s' (%s) for split '%s'",
            name,
            dataset_class.__name__,
            split,
        )

        return dataset_class(
            merged_config,
            split,
            transforms,
            model_preprocess,
            **init_kwargs,
        )

    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Failed to create dataset '{name}': {exc}") from exc
