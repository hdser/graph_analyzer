"""
Inference Module for GIT-CD

Provides inference utilities:
- EmbeddingPredictor: Batch inference for embeddings
- Similarity search
- Embedding visualization
- Community prediction
"""

from .predictor import (
    EmbeddingPredictor,
    SimilaritySearch,
    EmbeddingVisualizer,
)

__all__ = [
    "EmbeddingPredictor",
    "SimilaritySearch",
    "EmbeddingVisualizer",
]
