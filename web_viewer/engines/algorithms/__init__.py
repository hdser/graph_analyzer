"""
Anomaly Detection Algorithms Package

Provides various anomaly detection algorithms for use with the AnomalyEngine.

Available Algorithms:
- zscore: Z-Score based detection (statistical)
- modified_zscore: Modified Z-Score using MAD (statistical)
- iqr: Interquartile Range based detection (statistical)
- mahalanobis: Mahalanobis distance based detection (distance)
- isolation_forest: Isolation Forest (machine learning, requires sklearn)
- lof: Local Outlier Factor (density, requires sklearn)
- dbscan: DBSCAN clustering (density, requires sklearn)
- ocsvm: One-Class SVM (machine learning, requires sklearn)
- elliptic_envelope: Elliptic Envelope (statistical, requires sklearn)
"""

from typing import Dict, Any, Optional, Type

from .base import (
    BaseAnomalyAlgorithm,
    AlgorithmInfo,
    AlgorithmType,
    AlgorithmOutput,
    ParameterSpec,
)
from .statistical import (
    ZScoreAlgorithm,
    IQRAlgorithm,
    MahalanobisAlgorithm,
    ModifiedZScoreAlgorithm,
)

# Check for sklearn
try:
    from .sklearn_algorithms import (
        IsolationForestAlgorithm,
        LOFAlgorithm,
        DBSCANAlgorithm,
        OneClassSVMAlgorithm,
        EllipticEnvelopeAlgorithm,
        HAS_SKLEARN,
    )
except ImportError:
    HAS_SKLEARN = False
    IsolationForestAlgorithm = None
    LOFAlgorithm = None
    DBSCANAlgorithm = None
    OneClassSVMAlgorithm = None
    EllipticEnvelopeAlgorithm = None


# Algorithm registry
ALGORITHM_REGISTRY: Dict[str, Type[BaseAnomalyAlgorithm]] = {
    # Statistical (always available)
    "zscore": ZScoreAlgorithm,
    "modified_zscore": ModifiedZScoreAlgorithm,
    "iqr": IQRAlgorithm,
    "mahalanobis": MahalanobisAlgorithm,
}

# Add sklearn algorithms if available
if HAS_SKLEARN:
    ALGORITHM_REGISTRY.update({
        "isolation_forest": IsolationForestAlgorithm,
        "lof": LOFAlgorithm,
        "dbscan": DBSCANAlgorithm,
        "ocsvm": OneClassSVMAlgorithm,
        "elliptic_envelope": EllipticEnvelopeAlgorithm,
    })


def get_algorithm(name: str, **kwargs) -> BaseAnomalyAlgorithm:
    """
    Get an algorithm instance by name.
    
    Args:
        name: Algorithm name
        **kwargs: Algorithm parameters
        
    Returns:
        Algorithm instance
        
    Raises:
        ValueError: If algorithm not found
        ImportError: If sklearn not available
    """
    name_lower = name.lower()
    
    if name_lower not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
    
    algo_class = ALGORITHM_REGISTRY[name_lower]
    
    if algo_class is None:
        raise ImportError(f"Algorithm {name} requires sklearn which is not installed")
    
    return algo_class(**kwargs)


def list_algorithms() -> Dict[str, Any]:
    """
    List all available algorithms with their info.
    
    Returns:
        Dictionary mapping algorithm names to AlgorithmInfo dicts
    """
    result = {}
    
    for name, algo_class in ALGORITHM_REGISTRY.items():
        if algo_class is None:
            continue
        
        info = algo_class().get_info()
        # Convert to dict for JSON serialization
        result[name] = {
            "name": info.name,
            "display_name": info.display_name,
            "description": info.description,
            "algorithm_type": info.algorithm_type.value,
            "requires_sklearn": info.requires_sklearn,
            "supports_multivariate": info.supports_multivariate,
            "complexity": info.complexity,
            "max_recommended_nodes": info.max_recommended_nodes,
            "default_params": info.default_params,
            "param_descriptions": info.param_descriptions,
        }
    
    return result


def get_algorithm_info(name: str) -> Optional[AlgorithmInfo]:
    """
    Get information about a specific algorithm.
    
    Args:
        name: Algorithm name
        
    Returns:
        AlgorithmInfo or None if not found
    """
    name_lower = name.lower()
    
    if name_lower not in ALGORITHM_REGISTRY:
        return None
    
    algo_class = ALGORITHM_REGISTRY[name_lower]
    if algo_class is None:
        return None
    
    return algo_class().get_info()


__all__ = [
    # Base classes
    "BaseAnomalyAlgorithm",
    "AlgorithmInfo",
    "AlgorithmType",
    "AlgorithmOutput",
    "ParameterSpec",
    # Statistical algorithms
    "ZScoreAlgorithm",
    "IQRAlgorithm",
    "MahalanobisAlgorithm",
    "ModifiedZScoreAlgorithm",
    # Sklearn algorithms (may be None)
    "IsolationForestAlgorithm",
    "LOFAlgorithm",
    "DBSCANAlgorithm",
    "OneClassSVMAlgorithm",
    "EllipticEnvelopeAlgorithm",
    # Registry and functions
    "ALGORITHM_REGISTRY",
    "HAS_SKLEARN",
    "get_algorithm",
    "list_algorithms",
    "get_algorithm_info",
]