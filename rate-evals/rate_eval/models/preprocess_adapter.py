"""Picklable adapter wrapping a model class's `preprocess_single` staticmethod."""

from __future__ import annotations

from typing import Any


class ModelPreprocessor:
    """Picklable preprocessor for `mp.spawn` workers (wraps class + config)."""

    def __init__(self, model_cls: type, model_config: Any) -> None:
        self.model_cls = model_cls
        self.model_config = model_config

    def __call__(self, image: Any, **kwargs: Any) -> Any:
        preprocess_method = getattr(self.model_cls, "preprocess_single", None)
        if preprocess_method is None:
            raise AttributeError(
                f"Model {self.model_cls.__name__} does not have preprocess_single method"
            )
        return self.model_cls.preprocess_single(
            image, model_config=self.model_config, **kwargs
        )


__all__ = ["ModelPreprocessor"]
