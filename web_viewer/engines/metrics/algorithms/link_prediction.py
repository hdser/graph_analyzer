"""
Link Prediction Metric Algorithms

Algorithms for predicting missing links and measuring node similarity
based on network structure.
"""

from typing import Dict, Any, List
import logging
from collections import defaultdict

import networkx as nx

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class CommonNeighborsAlgorithm(BaseMetricAlgorithm):
    """
    Compute common neighbors score.
    
    For each node, computes the sum/average of common neighbors
    with all other nodes (as a measure of link prediction potential).
    """
    
    name = "common_neighbors"
    category = "link_prediction"
    description = "Common neighbors score for link prediction"
    cost = "medium"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {
                        "common_neighbors_sum": 0,
                        "common_neighbors_max": 0,
                    }
                    continue
                
                neighbors = set(U.neighbors(node))
                
                # Compute common neighbors with non-neighbors
                cn_scores = []
                for other in U.nodes():
                    if other != node and other not in neighbors:
                        # Common neighbors between node and other
                        other_neighbors = set(U.neighbors(other))
                        cn = len(neighbors & other_neighbors)
                        if cn > 0:
                            cn_scores.append(cn)
                
                result[node] = {
                    "common_neighbors_sum": sum(cn_scores) if cn_scores else 0,
                    "common_neighbors_max": max(cn_scores) if cn_scores else 0,
                }
            
            return result
        except Exception as e:
            logger.warning(f"Common neighbors failed: {e}")
            return {node: {} for node in nodes}


class PreferentialAttachmentAlgorithm(BaseMetricAlgorithm):
    """
    Compute preferential attachment score.
    
    For each node, computes the product of its degree with
    the average degree of non-neighbors (preferential attachment).
    """
    
    name = "preferential_attachment"
    category = "link_prediction"
    description = "Preferential attachment score"
    cost = "low"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        try:
            result = {}
            
            for node in nodes:
                if node not in U:
                    result[node] = {"preferential_attachment_score": 0}
                    continue
                
                k_node = U.degree(node)
                neighbors = set(U.neighbors(node))
                neighbors.add(node)  # Exclude self
                
                # Compute PA score as k_node * avg(k_non_neighbor)
                non_neighbor_degrees = []
                for other in U.nodes():
                    if other not in neighbors:
                        non_neighbor_degrees.append(U.degree(other))
                
                if non_neighbor_degrees:
                    avg_degree = sum(non_neighbor_degrees) / len(non_neighbor_degrees)
                    pa_score = k_node * avg_degree
                else:
                    pa_score = 0
                
                result[node] = {"preferential_attachment_score": pa_score}
            
            return result
        except Exception as e:
            logger.warning(f"Preferential attachment failed: {e}")
            return {node: {} for node in nodes}


class LinkPredictionScoresAlgorithm(BaseMetricAlgorithm):
    """
    Compute multiple link prediction scores.
    
    Computes Adamic-Adar, Resource Allocation, and Jaccard scores
    aggregated for each node.
    """
    
    name = "link_prediction_scores"
    category = "link_prediction"
    description = "Multiple link prediction scores (AA, RA, Jaccard)"
    cost = "high"
    graph_type = "undirected"
    # max_nodes is handled by registry/resolver - no internal limit
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, parameters=None, **kwargs) -> Dict[str, Dict[str, Any]]:
        params = parameters or {}
        top_k = params.get('top_k', 10)  # Number of top predictions to consider
        
        try:
            import math
            
            result = {}
            
            for node in nodes:
                if node not in U or U.degree(node) == 0:
                    result[node] = {
                        "aa_score_avg": 0,
                        "ra_score_avg": 0,
                        "jaccard_score_avg": 0,
                    }
                    continue
                
                neighbors = set(U.neighbors(node))
                
                aa_scores = []
                ra_scores = []
                jaccard_scores = []
                
                # Compute scores for non-neighbors
                for other in U.nodes():
                    if other == node or other in neighbors:
                        continue
                    
                    other_neighbors = set(U.neighbors(other))
                    common = neighbors & other_neighbors
                    
                    if len(common) == 0:
                        continue
                    
                    # Adamic-Adar Index
                    aa = 0
                    for cn in common:
                        degree_cn = U.degree(cn)
                        if degree_cn > 1:
                            aa += 1 / math.log(degree_cn)
                    aa_scores.append(aa)
                    
                    # Resource Allocation Index
                    ra = 0
                    for cn in common:
                        degree_cn = U.degree(cn)
                        if degree_cn > 0:
                            ra += 1 / degree_cn
                    ra_scores.append(ra)
                    
                    # Jaccard Coefficient
                    union = neighbors | other_neighbors
                    if len(union) > 0:
                        jaccard = len(common) / len(union)
                        jaccard_scores.append(jaccard)
                
                # Take top-k and compute averages
                aa_scores.sort(reverse=True)
                ra_scores.sort(reverse=True)
                jaccard_scores.sort(reverse=True)
                
                aa_top = aa_scores[:top_k]
                ra_top = ra_scores[:top_k]
                jaccard_top = jaccard_scores[:top_k]
                
                result[node] = {
                    "aa_score_avg": sum(aa_top) / len(aa_top) if aa_top else 0,
                    "ra_score_avg": sum(ra_top) / len(ra_top) if ra_top else 0,
                    "jaccard_score_avg": sum(jaccard_top) / len(jaccard_top) if jaccard_top else 0,
                }
            
            return result
        except Exception as e:
            logger.warning(f"Link prediction scores failed: {e}")
            return {node: {} for node in nodes}