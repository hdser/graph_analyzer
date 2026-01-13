"""
Topology Metric Algorithms

Basic degree and structure metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class InDegreeAlgorithm(BaseMetricAlgorithm):
    """Compute in-degree for each node."""
    
    name = "in_degree"
    category = "topology"
    description = "Number of incoming edges"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        for node in nodes:
            result[node] = {"in_degree": G.in_degree(node)}
        return result


class OutDegreeAlgorithm(BaseMetricAlgorithm):
    """Compute out-degree for each node."""
    
    name = "out_degree"
    category = "topology"
    description = "Number of outgoing edges"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        for node in nodes:
            result[node] = {"out_degree": G.out_degree(node)}
        return result


class TotalDegreeAlgorithm(BaseMetricAlgorithm):
    """Compute total degree (in + out) for each node."""
    
    name = "total_degree"
    category = "topology"
    description = "Sum of in and out degree"
    cost = "low"
    
    def compute(
        self, 
        G: nx.DiGraph, 
        U: nx.Graph, 
        nodes: list,
        computed_metrics: Dict[str, Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        result = {}
        for node in nodes:
            # Try to use pre-computed values
            if computed_metrics and node in computed_metrics:
                in_d = computed_metrics[node].get("in_degree", G.in_degree(node))
                out_d = computed_metrics[node].get("out_degree", G.out_degree(node))
            else:
                in_d = G.in_degree(node)
                out_d = G.out_degree(node)
            result[node] = {"total_degree": in_d + out_d}
        return result


class DegreeImbalanceAlgorithm(BaseMetricAlgorithm):
    """Compute degree imbalance (normalized difference between in and out)."""
    
    name = "degree_imbalance"
    category = "topology"
    description = "Normalized difference between in and out degree"
    cost = "low"
    
    def compute(
        self,
        G: nx.DiGraph,
        U: nx.Graph,
        nodes: list,
        computed_metrics: Dict[str, Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        result = {}
        for node in nodes:
            # Try to use pre-computed values
            if computed_metrics and node in computed_metrics:
                in_d = computed_metrics[node].get("in_degree", G.in_degree(node))
                out_d = computed_metrics[node].get("out_degree", G.out_degree(node))
            else:
                in_d = G.in_degree(node)
                out_d = G.out_degree(node)
            
            total = in_d + out_d
            if total > 0:
                imbalance = abs(in_d - out_d) / total
            else:
                imbalance = 0.0
            
            result[node] = {"degree_imbalance": imbalance}
        return result