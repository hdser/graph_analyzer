"""
Deep Learning Module for Graph Analyzer

Provides GIT-CD (Graph Integrated Transformer for Community Detection)
and related deep learning capabilities for:
- Node embedding generation
- Advanced community detection
- Semi-supervised node classification
- Embedding-based anomaly detection

This module is optional. If PyTorch/PyG are not installed,
the system gracefully falls back to traditional algorithms.
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Check for PyTorch availability
HAS_TORCH = False
TORCH_VERSION: Optional[str] = None
TORCH_CUDA_AVAILABLE = False

try:
    import torch
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    logger.info(f"PyTorch {TORCH_VERSION} available (CUDA: {TORCH_CUDA_AVAILABLE})")
except ImportError:
    logger.info("PyTorch not available - deep learning features disabled")

# Check for PyTorch Geometric availability
HAS_PYG = False
PYG_VERSION: Optional[str] = None

try:
    import torch_geometric
    HAS_PYG = True
    PYG_VERSION = torch_geometric.__version__
    logger.info(f"PyTorch Geometric {PYG_VERSION} available")
except ImportError:
    logger.info("PyTorch Geometric not available - GNN features disabled")

# Check for UMAP (embedding visualization)
HAS_UMAP = False
try:
    import umap
    HAS_UMAP = True
    logger.info("UMAP available for embedding visualization")
except ImportError:
    logger.info("UMAP not available - embedding visualization limited")

# Overall deep learning availability
HAS_DEEP_LEARNING = HAS_TORCH and HAS_PYG


def get_deep_learning_info() -> Dict[str, Any]:
    """
    Get information about deep learning availability.
    
    Returns:
        Dict with availability status and versions
    """
    return {
        "available": HAS_DEEP_LEARNING,
        "torch": {
            "available": HAS_TORCH,
            "version": TORCH_VERSION,
            "cuda_available": TORCH_CUDA_AVAILABLE,
            "cuda_device_count": (
                torch.cuda.device_count() if HAS_TORCH and TORCH_CUDA_AVAILABLE else 0
            ),
        },
        "torch_geometric": {
            "available": HAS_PYG,
            "version": PYG_VERSION,
        },
        "umap": {
            "available": HAS_UMAP,
        },
        "features": {
            "gitcd": HAS_DEEP_LEARNING,
            "node_embeddings": HAS_DEEP_LEARNING,
            "gnn_community_detection": HAS_DEEP_LEARNING,
            "embedding_visualization": HAS_UMAP,
        }
    }


# Conditional imports - only if dependencies available
if HAS_DEEP_LEARNING:
    from .config import DeepLearningConfig, GITCDConfig, TrainingConfig
    from .utils.feature_builder import FeatureBuilder, FeatureConfig
    from .utils.graph_converter import GraphConverter
    
    # Models
    from .models.gnn import HomogeneousSAGE, HeterogeneousSAGE
    from .models.transformer import DynamicHINAttention, HINTransformerBlock
    from .models.clustering_head import DeepClusteringHead
    from .models.gitcd import GITCD
    
    # Training
    from .training.trainer import GITCDTrainer, TrainingCallback
    from .training.losses import kl_divergence_loss, soft_silhouette_loss
    
    # Inference
    from .inference.predictor import EmbeddingPredictor, SimilaritySearch, EmbeddingVisualizer
    
    __all__ = [
        # Availability
        "HAS_DEEP_LEARNING",
        "HAS_TORCH",
        "HAS_PYG",
        "HAS_UMAP",
        "get_deep_learning_info",
        # Config
        "DeepLearningConfig",
        "GITCDConfig",
        "TrainingConfig",
        "FeatureConfig",
        # Utils
        "FeatureBuilder",
        "GraphConverter",
        # Models
        "HomogeneousSAGE",
        "HeterogeneousSAGE",
        "DynamicHINAttention",
        "HINTransformerBlock",
        "DeepClusteringHead",
        "GITCD",
        # Training
        "GITCDTrainer",
        "TrainingCallback",
        "kl_divergence_loss",
        "soft_silhouette_loss",
        # Inference
        "EmbeddingPredictor",
        "SimilaritySearch",
        "EmbeddingVisualizer",
    ]
else:
    __all__ = [
        "HAS_DEEP_LEARNING",
        "HAS_TORCH",
        "HAS_PYG",
        "HAS_UMAP",
        "get_deep_learning_info",
    ]