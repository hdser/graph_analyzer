"""
Flow Metric Algorithms

Flow and hierarchy metrics.
"""

from typing import Dict, Any
import logging

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class FlowHierarchyAlgorithm(BaseMetricAlgorithm):
    """Compute flow hierarchy of directed graph."""
    
    name = "flow_hierarchy"
    category = "flow"
    description = "Flow hierarchy of directed graph"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            if G.number_of_nodes() > 1:
                flow_h = nx.flow_hierarchy(G)
                logger.debug(f"Flow hierarchy: {flow_h:.4f}")
            else:
                flow_h = 0
            
            result = {}
            for node in nodes:
                result[node] = {"flow_hierarchy": flow_h}
            return result
        except Exception as e:
            logger.warning(f"Flow hierarchy failed: {e}")
            return {node: {} for node in nodes}