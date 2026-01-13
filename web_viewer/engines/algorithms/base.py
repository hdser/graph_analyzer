"""
Anomaly Detection Algorithm Base

Base classes and interfaces for anomaly detection algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

import numpy as np


class AlgorithmType(Enum):
    """Types of anomaly detection algorithms."""
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    DENSITY_BASED = "density_based"
    DISTANCE_BASED = "distance_based"


@dataclass
class ParameterSpec:
    """Specification for an algorithm parameter."""
    name: str
    param_type: str  # "float", "int", "str", "bool"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""


@dataclass
class AlgorithmInfo:
    """Information about an anomaly detection algorithm."""
    name: str
    display_name: str
    description: str
    algorithm_type: AlgorithmType
    requires_sklearn: bool = False
    supports_multivariate: bool = True
    complexity: str = "O(n)"
    max_recommended_nodes: Optional[int] = None
    supports_incremental: bool = False
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_descriptions: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlgorithmOutput:
    """Output from anomaly detection algorithm."""
    raw_scores: np.ndarray  # Anomaly scores (higher = more anomalous)
    anomaly_mask: np.ndarray  # Boolean mask (True = anomaly)
    threshold_used: float  # Threshold value used
    per_metric_scores: Optional[Dict[str, np.ndarray]] = None  # Per-feature scores


class BaseAnomalyAlgorithm(ABC):
    """
    Abstract base class for all anomaly detection algorithms.
    
    Algorithms compute anomaly scores for each data point.
    Higher scores indicate more anomalous points.
    """
    
    name: str = ""
    display_name: str = ""
    description: str = ""
    algorithm_type: AlgorithmType = AlgorithmType.STATISTICAL
    requires_sklearn: bool = False
    supports_multivariate: bool = True
    complexity: str = "O(n)"
    max_recommended_nodes: Optional[int] = None
    
    def __init__(self, **kwargs):
        """Initialize algorithm with parameters."""
        self.params = kwargs
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> AlgorithmOutput:
        """
        Compute anomaly scores for data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            params: Algorithm parameters
            
        Returns:
            AlgorithmOutput with scores, mask, and threshold
        """
        pass
    
    def validate_params(self, parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and fill in default parameters.
        
        Args:
            parameters: User-provided parameters (may be None)
            
        Returns:
            Complete parameter dict with defaults filled in
        """
        defaults = self.get_default_params()
        if parameters is None:
            return defaults
        
        # Merge with defaults
        result = defaults.copy()
        result.update(parameters)
        return result
    
    def get_info(self) -> AlgorithmInfo:
        """Get algorithm information."""
        return AlgorithmInfo(
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            algorithm_type=self.algorithm_type,
            requires_sklearn=self.requires_sklearn,
            supports_multivariate=self.supports_multivariate,
            complexity=self.complexity,
            max_recommended_nodes=self.max_recommended_nodes,
            parameters=self.get_parameter_specs(),
            default_params=self.get_default_params(),
            param_descriptions=self.get_param_descriptions(),
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get default parameters for this algorithm."""
        return {}
    
    @classmethod
    def get_param_descriptions(cls) -> Dict[str, str]:
        """Get parameter descriptions."""
        return {}
    
    @classmethod
    def get_parameter_specs(cls) -> Dict[str, ParameterSpec]:
        """Get full parameter specifications."""
        specs = {}
        defaults = cls.get_default_params()
        descriptions = cls.get_param_descriptions()
        
        for name, default in defaults.items():
            if isinstance(default, bool):
                param_type = "bool"
            elif isinstance(default, int):
                param_type = "int"
            elif isinstance(default, float):
                param_type = "float"
            elif isinstance(default, str):
                param_type = "str"
            else:
                param_type = "any"
            
            specs[name] = ParameterSpec(
                name=name,
                param_type=param_type,
                default=default,
                description=descriptions.get(name, ""),
            )
        
        return specs
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"