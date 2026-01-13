"""
Base Classes for Graph Analyzer

Contains base classes for:
- Anomaly detection algorithms (BaseAnomalyAlgorithm)
- Metric computation algorithms (BaseMetricAlgorithm)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from enum import Enum

import numpy as np
import networkx as nx

if TYPE_CHECKING:
    import igraph

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# ANOMALY DETECTION BASE CLASSES
# =============================================================================

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


# =============================================================================
# METRIC ALGORITHM BASE CLASS
# =============================================================================

class BaseMetricAlgorithm(ABC):
    """
    Abstract base class for all metric computation algorithms.
    
    All metric algorithms must inherit from this class and implement
    the compute() method.
    
    Class Attributes:
        name: Unique identifier for the algorithm
        category: Category this algorithm belongs to
        description: Human-readable description
        cost: Computational cost ("low", "medium", "high", "very_high")
        max_nodes: Maximum graph size this algorithm can handle (None = unlimited)
        graph_type: Type of graph needed ("directed", "undirected", "both")
        requires_connected: Whether the graph must be connected
        preferred_library: Preferred computation library ("networkx", "igraph", "both")
        requires_igraph: Whether igraph is required for this algorithm
    """
    
    name: str = ""
    category: str = ""
    description: str = ""
    cost: str = "low"
    max_nodes: Optional[int] = None
    graph_type: str = "both"
    requires_connected: bool = False
    preferred_library: str = "networkx"
    requires_igraph: bool = False
    
    def __init__(self, **kwargs):
        """Initialize algorithm with optional parameters."""
        self.params = kwargs
    
    @abstractmethod
    def compute(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        nodes: List[str],
        ig: Optional['igraph.Graph'] = None,
        parameters: Optional[Dict[str, Any]] = None,
        computed_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
        converters: Optional[List[str]] = None,
        n_jobs: int = 1,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute metric values for all nodes.
        
        Args:
            G: NetworkX directed graph
            U: NetworkX undirected version of graph
            nodes: List of node IDs to compute metrics for
            ig: Optional igraph version of graph (for igraph algorithms)
            parameters: Algorithm-specific parameters from user
            computed_metrics: Previously computed metrics (for dependencies)
            converters: Trusted seed nodes (for trust algorithms)
            n_jobs: Number of parallel workers
            **kwargs: Additional parameters
            
        Returns:
            Dict mapping node ID to dict of metric values
            Example: {"node1": {"metric_a": 0.5, "metric_b": 10}, ...}
        """
        pass
    
    def can_compute(self, n_nodes: int, is_connected: bool) -> bool:
        """
        Check if this algorithm can be computed for the given graph.
        
        Args:
            n_nodes: Number of nodes in graph
            is_connected: Whether graph is connected
            
        Returns:
            True if algorithm can be computed
        """
        # Check node limit
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            return False
        
        # Check connectivity requirement
        if self.requires_connected and not is_connected:
            return False
        
        return True
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this algorithm."""
        return {}
    
    @staticmethod
    def nx_to_igraph(G: nx.DiGraph) -> Optional['igraph.Graph']:
        """
        Convert NetworkX graph to igraph.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            igraph.Graph equivalent or None if igraph not available
        """
        try:
            import igraph as ig
            
            nodes = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
            
            g = ig.Graph(n=len(nodes), edges=edges, directed=True)
            g.vs['name'] = nodes
            
            # Copy edge weights if present
            if nx.is_weighted(G):
                weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
                g.es['weight'] = weights
            
            return g
        except ImportError:
            logger.warning("igraph not available")
            return None
    
    @staticmethod
    def nx_undirected_to_igraph(U: nx.Graph) -> Optional['igraph.Graph']:
        """
        Convert NetworkX undirected graph to igraph.
        
        Args:
            U: NetworkX undirected graph
            
        Returns:
            igraph.Graph equivalent or None if igraph not available
        """
        try:
            import igraph as ig
            
            nodes = list(U.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            edges = [(node_to_idx[u], node_to_idx[v]) for u, v in U.edges()]
            
            g = ig.Graph(n=len(nodes), edges=edges, directed=False)
            g.vs['name'] = nodes
            
            # Copy edge weights if present
            if nx.is_weighted(U):
                weights = [U[u][v].get('weight', 1.0) for u, v in U.edges()]
                g.es['weight'] = weights
            
            return g
        except ImportError:
            logger.warning("igraph not available")
            return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, cost={self.cost})"