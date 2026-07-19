"""Exception hierarchy for the RATE evaluation pipeline."""


class RATEEvalError(Exception):
    """Base exception for the RATE evaluation pipeline."""

    pass


class ModelError(RATEEvalError):
    """Raised when there's a model-related error."""

    pass


class DatasetError(RATEEvalError):
    """Raised when there's a dataset-related error."""

    pass
