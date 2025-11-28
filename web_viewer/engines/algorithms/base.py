"""
Base Classes for Anomaly Detection Algorithms

Defines the interface all algorithms must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum

import numpy as np


@dataclass
class ParameterSpec:
    """Specification for an algorithm parameter."""
    name: str
    param_type: str  # "float", "int", "bool", "str"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.param_type,
            'default': self.default,
            'min': self.min_value,
            'max': self.max_value,
            'choices': self.choices,
            'description': self.description,
        }
    
    def validate(self, value: Any) -> Any:
        """Validate and convert parameter value."""
        if value is None:
            return self.default
        
        # Type conversion
        if self.param_type == "float":
            value = float(value)
        elif self.param_type == "int":
            value = int(value)
        elif self.param_type == "bool":
            value = bool(value)
        elif self.param_type == "str":
            value = str(value)
        
        # Range check
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"Parameter {self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"Parameter {self.name} must be <= {self.max_value}")
        
        # Choice check
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"Parameter {self.name} must be one of {self.choices}")
        
        return value


@dataclass
class AlgorithmInfo:
    """Information about an algorithm."""
    name: str
    display_name: str
    description: str
    complexity: str
    supports_multivariate: bool
    requires_sklearn: bool
    parameters: Dict[str, ParameterSpec]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'complexity': self.complexity,
            'multivariate': self.supports_multivariate,
            'requires_sklearn': self.requires_sklearn,
            'parameters': {
                name: spec.to_dict()
                for name, spec in self.parameters.items()
            },
        }


@dataclass
class AlgorithmOutput:
    """Output from an algorithm's fit_predict method."""
    raw_scores: np.ndarray
    anomaly_mask: np.ndarray
    threshold_used: float
    per_metric_scores: Optional[Dict[str, np.ndarray]] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)


class AnomalyAlgorithmBase(ABC):
    """
    Base class for all anomaly detection algorithms.
    
    Subclasses must implement:
    - fit_predict(): Run detection and return scores/labels
    - get_info(): Return algorithm metadata
    
    All algorithms are metric-agnostic - they work with 
    a preprocessed data matrix X.
    """
    
    # Class attributes to be overridden by subclasses
    name: str = "base"
    display_name: str = "Base Algorithm"
    description: str = "Base anomaly detection algorithm"
    complexity: str = "O(n)"
    supports_multivariate: bool = True
    requires_sklearn: bool = False
    
    # Parameter specifications
    parameters: Dict[str, ParameterSpec] = {}
    
    def __init__(self):
        pass
    
    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        """Get algorithm information."""
        return AlgorithmInfo(
            name=cls.name,
            display_name=cls.display_name,
            description=cls.description,
            complexity=cls.complexity,
            supports_multivariate=cls.supports_multivariate,
            requires_sklearn=cls.requires_sklearn,
            parameters=cls.parameters,
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get default parameter values."""
        return {
            name: spec.default
            for name, spec in cls.parameters.items()
        }
    
    def validate_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate and fill in default parameters.
        
        Args:
            params: User-provided parameters (may be None or partial)
            
        Returns:
            Complete parameter dictionary with defaults filled in
        """
        result = self.get_default_params()
        
        if params:
            for name, value in params.items():
                if name in self.parameters:
                    result[name] = self.parameters[name].validate(value)
        
        return result
    
    @abstractmethod
    def fit_predict(
        self,
        X: np.ndarray,
        params: Dict[str, Any],
    ) -> AlgorithmOutput:
        """
        Run anomaly detection.
        
        Args:
            X: Data matrix (n_samples, n_features), preprocessed
            params: Algorithm parameters (validated)
            
        Returns:
            AlgorithmOutput with scores, labels, and metadata
        """
        pass
    
    def _validate_input(self, X: np.ndarray) -> None:
        """Validate input data."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if X.shape[0] == 0:
            raise ValueError("X has no samples")
        
        if X.shape[1] == 0:
            raise ValueError("X has no features")
        
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values after preprocessing")
        
        if np.any(np.isinf(X)):
            raise ValueError("X contains infinite values after preprocessing")