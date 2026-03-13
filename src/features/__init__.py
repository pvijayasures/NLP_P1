"""
Feature engineering module.

Provides a unified interface for creating text features
independent of the underlying method (TF-IDF, embeddings).
"""

from .feature_pipeline import prepare_features

__all__ = ["prepare_features"]