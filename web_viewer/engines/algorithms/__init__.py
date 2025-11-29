"""
Anomaly Detection Algorithms

Registry and exports for all anomaly detection algorithms.
"""

from typing import Dict, Type, Any, Optional
import warnings

from .base import (
    AnomalyAlgorithmBase,
    ParameterSpec,
    AlgorithmInfo,
    AlgorithmOutput,
)
from .statistical import ZScoreAlgorithm, IQRAlgorithm
from .distance_based import MahalanobisAlgorithm

# Check sklearn availability
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not installed. ML-based algorithms unavailable.")

# Import ML algorithms only if sklearn available
if HAS_SKLEARN:
    from .ml_based import (
        IsolationForestAlgorithm,
        LOFAlgorithm,
        DBSCANAlgorithm,
        PCAReconstructionAlgorithm,
        OneClassSVMAlgorithm,
    )


# Algorithm registry
ALGORITHM_REGISTRY: Dict[str, Type[AnomalyAlgorithmBase]] = {
    "zscore": ZScoreAlgorithm,
    "iqr": IQRAlgorithm,
    "mahalanobis": MahalanobisAlgorithm,
}

if HAS_SKLEARN:
    ALGORITHM_REGISTRY.update({
        "isolation_forest": IsolationForestAlgorithm,
        "lof": LOFAlgorithm,
        "dbscan": DBSCANAlgorithm,
        "pca_reconstruction": PCAReconstructionAlgorithm,
        "one_class_svm": OneClassSVMAlgorithm,
    })


def get_algorithm(name: str) -> AnomalyAlgorithmBase:
    """
    Get algorithm instance by name.
    
    Args:
        name: Algorithm name
        
    Returns:
        Algorithm instance
        
    Raises:
        ValueError: If algorithm not found
    """
    if name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
    
    return ALGORITHM_REGISTRY[name]()


def list_algorithms() -> Dict[str, AlgorithmInfo]:
    """
    List all available algorithms with their info.
    
    Returns:
        Dictionary mapping algorithm name to info
    """
    return {
        name: algo_class.get_info()
        for name, algo_class in ALGORITHM_REGISTRY.items()
    }


def get_algorithm_info(name: str) -> AlgorithmInfo:
    """
    Get detailed info for a specific algorithm.
    
    Args:
        name: Algorithm name
        
    Returns:
        Algorithm info
    """
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}")
    
    return ALGORITHM_REGISTRY[name].get_info()


__all__ = [
    # Base classes
    "AnomalyAlgorithmBase",
    "ParameterSpec",
    "AlgorithmInfo",
    "AlgorithmOutput",
    # Registry functions
    "get_algorithm",
    "list_algorithms",
    "get_algorithm_info",
    "ALGORITHM_REGISTRY",
    "HAS_SKLEARN",
    # Statistical algorithms
    "ZScoreAlgorithm",
    "IQRAlgorithm",
    # Distance-based algorithms
    "MahalanobisAlgorithm",
]

if HAS_SKLEARN:
    __all__.extend([
        # ML-based algorithms
        "IsolationForestAlgorithm",
        "LOFAlgorithm",
        "DBSCANAlgorithm",
        "PCAReconstructionAlgorithm",
        "OneClassSVMAlgorithm",
    ])