"""
Bipartite Metric Algorithms

Metrics specific to bipartite graphs or bipartite subgraphs.
"""

from typing import Dict, Any, List
import logging

import networkx as nx
from networkx.algorithms import bipartite

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class BipartiteProjectionDegreeAlgorithm(BaseMetricAlgorithm):
    """
    Compute bipartite projection degree.
    
    If the graph has a bipartite structure, computes metrics related
    to the bipartite projection.
    """
    
    name = "bipartite_projection_degree"
    category = "bipartite"
    description = "Bipartite projection degree and redundancy"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            # Check if graph is bipartite
            if not bipartite.is_bipartite(U):
                # Graph is not bipartite, compute based on 2-hop connectivity
                result = {}
                for node in nodes:
                    if node not in U:
                        result[node] = {
                            "bipartite_projection_degree": 0,
                            "bipartite_redundancy": 0,
                        }
                        continue
                    
                    # Count 2-hop neighbors (as proxy for projection)
                    neighbors = set(U.neighbors(node))
                    two_hop = set()
                    for n in neighbors:
                        two_hop.update(U.neighbors(n))
                    two_hop.discard(node)
                    two_hop -= neighbors
                    
                    result[node] = {
                        "bipartite_projection_degree": len(two_hop),
                        "bipartite_redundancy": 0,
                    }
                
                return result
            
            # Graph is bipartite, get the two sets
            top_nodes, bottom_nodes = bipartite.sets(U)
            
            # Compute redundancy coefficient
            try:
                redundancy = bipartite.node_redundancy(U)
            except Exception:
                redundancy = {n: 0 for n in U.nodes()}
            
            result = {}
            for node in nodes:
                if node not in U:
                    result[node] = {
                        "bipartite_projection_degree": 0,
                        "bipartite_redundancy": 0,
                    }
                    continue
                
                # Get projection degree (2-hop neighbors in same partition)
                neighbors = set(U.neighbors(node))
                two_hop = set()
                for n in neighbors:
                    two_hop.update(U.neighbors(n))
                two_hop.discard(node)
                
                result[node] = {
                    "bipartite_projection_degree": len(two_hop),
                    "bipartite_redundancy": redundancy.get(node, 0),
                }
            
            return result
        except Exception as e:
            logger.warning(f"Bipartite projection degree failed: {e}")
            return {node: {} for node in nodes}