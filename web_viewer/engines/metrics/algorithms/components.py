"""
Component Metric Algorithms

Component membership metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class ComponentsAlgorithm(BaseMetricAlgorithm):
    """Compute weak and strong component membership."""
    
    name = "components"
    category = "components"
    description = "Weak and strong component membership"
    cost = "low"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            weak_comps = list(nx.weakly_connected_components(G))
            strong_comps = list(nx.strongly_connected_components(G))
            
            logger.debug(f"Found {len(weak_comps)} weak components, {len(strong_comps)} strong components")
            
            # Map nodes to component IDs and sizes
            weak_map = {}
            weak_sizes = {}
            for idx, comp in enumerate(weak_comps):
                weak_sizes[idx] = len(comp)
                for node in comp:
                    weak_map[node] = idx
            
            strong_map = {}
            strong_sizes = {}
            for idx, comp in enumerate(strong_comps):
                strong_sizes[idx] = len(comp)
                for node in comp:
                    strong_map[node] = idx
            
            largest_weak = max(weak_sizes.values()) if weak_sizes else 0
            
            result = {}
            for node in nodes:
                wid = weak_map.get(node, -1)
                sid = strong_map.get(node, -1)
                result[node] = {
                    "weak_component_size": weak_sizes.get(wid, 0),
                    "strong_component_size": strong_sizes.get(sid, 0),
                    "in_largest_component": 1 if weak_sizes.get(wid, 0) == largest_weak else 0,
                }
            return result
        except Exception as e:
            logger.warning(f"Component metrics failed: {e}")
            return {node: {} for node in nodes}