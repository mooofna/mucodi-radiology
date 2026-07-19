"""Command-line interface for the RATE evaluation pipeline."""

from .extract import extract_embeddings_cli
from .evaluate import evaluate_embeddings_cli
from .deploy import deploy_embeddings_cli

__all__ = ["extract_embeddings_cli", "evaluate_embeddings_cli", "deploy_embeddings_cli"]
