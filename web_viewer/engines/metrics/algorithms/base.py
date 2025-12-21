"""
Base Metric Algorithm

Abstract base class for all metric algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

import networkx as nx

logger = logging.getLogger(__name__)


class BaseMetricAlgorithm(ABC):
    """
    Base class for all metric algorithms.
    
    Each algorithm computes one or more metrics for all nodes in a graph.
    Subclasses must implement the `compute` method.
    
    Attributes:
        name: Unique identifier for this algorithm
        category: Category this algorithm belongs to
        description: Human-readable description
        cost: Computational cost ("low", "medium", "high", "very_high")
        max_nodes: Maximum graph size this algorithm supports (None = unlimited)
        graph_type: Required graph type ("directed", "undirected", "both")
        requires_connected: Whether algorithm requires connected graph
    """
    
    name: str = ""
    category: str = ""
    description: str = ""
    cost: str = "low"
    max_nodes: Optional[int] = None
    graph_type: str = "directed"
    requires_connected: bool = False
    
    def __init__(self, **kwargs):
        """Initialize algorithm with optional parameters."""
        self.params = kwargs
    
    @abstractmethod
    def compute(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        nodes: list,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute metric(s) for all nodes.
        
        Args:
            G: Directed graph
            U: Undirected version of the graph
            nodes: List of node IDs to compute for
            **kwargs: Additional parameters (converters, trust_matrix, etc.)
            
        Returns:
            Dict mapping node_id -> {metric_name: value, ...}
            
        Example:
            {
                "node_1": {"pagerank": 0.05, "pagerank_undirected": 0.04},
                "node_2": {"pagerank": 0.03, "pagerank_undirected": 0.03},
            }
        """
        pass
    
    def can_compute(self, n_nodes: int, is_connected: bool = True) -> bool:
        """
        Check if this algorithm can be computed for the given graph.
        
        Args:
            n_nodes: Number of nodes in the graph
            is_connected: Whether the graph is connected
            
        Returns:
            True if algorithm can be computed
        """
        if self.max_nodes is not None and n_nodes > self.max_nodes:
            logger.debug(f"{self.name}: Skipping, graph too large ({n_nodes} > {self.max_nodes})")
            return False
        
        if self.requires_connected and not is_connected:
            logger.debug(f"{self.name}: Skipping, requires connected graph")
            return False
        
        return True
    
    def get_empty_result(self, nodes: list) -> Dict[str, Dict[str, Any]]:
        """Return empty result structure for when computation is skipped."""
        return {node: {} for node in nodes}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, cost={self.cost})"