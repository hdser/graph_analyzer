"""
Utilities Module for GIT-CD

Provides utility classes for:
- Feature building and normalization
- Graph conversion (NetworkX → PyTorch Geometric)
- Embedding caching
"""

from .feature_builder import (
    FeatureBuilder,
    FeatureStats,
)

from .graph_converter import (
    GraphConverter,
    ConversionConfig,
    create_train_val_test_masks,
    create_label_masks,
)

from .cache import (
    EmbeddingCache,
    compute_graph_hash,
    compute_config_hash,
)

__all__ = [
    # Feature building
    "FeatureBuilder",
    "FeatureStats",
    
    # Graph conversion
    "GraphConverter",
    "ConversionConfig",
    "create_train_val_test_masks",
    "create_label_masks",
    
    # Caching
    "EmbeddingCache",
    "compute_graph_hash",
    "compute_config_hash",
]
