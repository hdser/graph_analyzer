"""
Influence Metric Algorithms

Influence maximization and spreading algorithms.
"""

from typing import Dict, Any, List
import logging
from collections import defaultdict

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class CollectiveInfluenceAlgorithm(BaseMetricAlgorithm):
    """
    Compute collective influence.
    
    Collective influence (CI) measures the importance of a node for
    network dismantling. Based on the Morone & Makse algorithm.
    """
    
    name = "collective_influence"
    category = "influence"
    description = "Collective influence for network dismantling"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        l = params.get('ball_radius', 2)  # Ball radius for CI computation
        
        try:
            result = {}
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {"collective_influence": 0}
                    continue
                
                k_i = U.degree(node)
                
                # Find nodes at distance exactly l
                ball_boundary = set()
                try:
                    lengths = nx.single_source_shortest_path_length(U, node, cutoff=l)
                    for n, dist in lengths.items():
                        if dist == l:
                            ball_boundary.add(n)
                except Exception:
                    pass
                
                # CI = (k_i - 1) * sum(k_j - 1) for j in ball boundary
                ci = 0
                if ball_boundary:
                    boundary_sum = sum(U.degree(n) - 1 for n in ball_boundary)
                    ci = (k_i - 1) * boundary_sum
                
                result[node] = {"collective_influence": ci}
            
            return result
        except Exception as e:
            logger.warning(f"Collective influence failed: {e}")
            return {node: {} for node in nodes}


class SpreadingActivationAlgorithm(BaseMetricAlgorithm):
    """
    Compute spreading activation potential.
    
    Simulates information spreading from each node to measure
    its potential to influence the network.
    """
    
    name = "spreading_activation"
    category = "influence"
    description = "Spreading activation influence potential"
    cost = "medium"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        steps = params.get('steps', 3)  # Number of spreading steps
        decay = params.get('decay', 0.5)  # Decay factor per step
        
        try:
            result = {}
            
            for node in nodes:
                if node not in G:
                    result[node] = {
                        "spreading_activation": 0,
                        "influence_reach": 0,
                    }
                    continue
                
                # Initialize activation
                activation = {n: 0.0 for n in G.nodes()}
                activation[node] = 1.0
                
                # Spread activation
                for step in range(steps):
                    new_activation = {n: 0.0 for n in G.nodes()}
                    for n in G.nodes():
                        if activation[n] > 0:
                            # Spread to successors
                            successors = list(G.successors(n))
                            if successors:
                                spread = activation[n] * decay / len(successors)
                                for s in successors:
                                    new_activation[s] += spread
                    
                    # Update activation (keep some of original)
                    for n in G.nodes():
                        activation[n] = activation[n] * decay + new_activation[n]
                
                # Total influence is sum of all activations
                total_activation = sum(activation.values())
                influenced_count = sum(1 for a in activation.values() if a > 0.01)
                
                result[node] = {
                    "spreading_activation": total_activation,
                    "influence_reach": influenced_count,
                }
            
            return result
        except Exception as e:
            logger.warning(f"Spreading activation failed: {e}")
            return {node: {} for node in nodes}