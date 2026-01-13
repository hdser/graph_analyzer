"""
Similarity Metric Algorithms

Node similarity measures.
"""

from typing import Dict, Any
import logging

import networkx as nx
import numpy as np

from .base import BaseMetricAlgorithm

logger = logging.getLogger(__name__)


class JaccardSimilarityAlgorithm(BaseMetricAlgorithm):
    """Compute average Jaccard similarity with neighbors."""
    
    name = "jaccard_similarity"
    category = "similarity"
    description = "Average Jaccard similarity with neighbors"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            neighbors = list(U.neighbors(node))
            if not neighbors:
                result[node] = {"jaccard_similarity_avg": 0.0, "jaccard_similarity_max": 0.0}
                continue
            
            similarities = []
            node_neighbors = set(U.neighbors(node))
            
            for neighbor in neighbors:
                neighbor_neighbors = set(U.neighbors(neighbor))
                intersection = len(node_neighbors & neighbor_neighbors)
                union = len(node_neighbors | neighbor_neighbors)
                if union > 0:
                    similarities.append(intersection / union)
            
            if similarities:
                result[node] = {
                    "jaccard_similarity_avg": float(np.mean(similarities)),
                    "jaccard_similarity_max": float(np.max(similarities)),
                }
            else:
                result[node] = {"jaccard_similarity_avg": 0.0, "jaccard_similarity_max": 0.0}
        
        return result


class CosineSimilarityAlgorithm(BaseMetricAlgorithm):
    """Compute average cosine similarity with neighbors based on neighbor overlap."""
    
    name = "cosine_similarity"
    category = "similarity"
    description = "Average cosine similarity with neighbors"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            neighbors = list(U.neighbors(node))
            if not neighbors:
                result[node] = {"cosine_similarity_avg": 0.0, "cosine_similarity_max": 0.0}
                continue
            
            similarities = []
            node_neighbors = set(U.neighbors(node))
            node_degree = len(node_neighbors)
            
            for neighbor in neighbors:
                neighbor_neighbors = set(U.neighbors(neighbor))
                neighbor_degree = len(neighbor_neighbors)
                intersection = len(node_neighbors & neighbor_neighbors)
                
                denominator = np.sqrt(node_degree * neighbor_degree)
                if denominator > 0:
                    similarities.append(intersection / denominator)
            
            if similarities:
                result[node] = {
                    "cosine_similarity_avg": float(np.mean(similarities)),
                    "cosine_similarity_max": float(np.max(similarities)),
                }
            else:
                result[node] = {"cosine_similarity_avg": 0.0, "cosine_similarity_max": 0.0}
        
        return result


class AdamicAdarAlgorithm(BaseMetricAlgorithm):
    """
    Compute Adamic-Adar similarity scores.
    
    Adamic-Adar index gives weight to common neighbors by the inverse
    log of their degree - emphasizing low-degree common neighbors.
    """
    
    name = "adamic_adar"
    category = "similarity"
    description = "Adamic-Adar similarity scores"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            neighbors = list(U.neighbors(node))
            if not neighbors:
                result[node] = {"adamic_adar_sum": 0.0, "adamic_adar_avg": 0.0}
                continue
            
            scores = []
            node_neighbors = set(U.neighbors(node))
            
            for neighbor in neighbors:
                neighbor_neighbors = set(U.neighbors(neighbor))
                common = node_neighbors & neighbor_neighbors
                
                aa_score = 0.0
                for common_neighbor in common:
                    degree = U.degree(common_neighbor)
                    if degree > 1:
                        aa_score += 1.0 / np.log(degree)
                
                scores.append(aa_score)
            
            if scores:
                result[node] = {
                    "adamic_adar_sum": float(np.sum(scores)),
                    "adamic_adar_avg": float(np.mean(scores)),
                }
            else:
                result[node] = {"adamic_adar_sum": 0.0, "adamic_adar_avg": 0.0}
        
        return result


class ResourceAllocationAlgorithm(BaseMetricAlgorithm):
    """
    Compute Resource Allocation index.
    
    Similar to Adamic-Adar but uses 1/degree instead of 1/log(degree),
    based on the idea of resource allocation through common neighbors.
    """
    
    name = "resource_allocation"
    category = "similarity"
    description = "Resource allocation index"
    cost = "high"
    graph_type = "undirected"
    
    def compute(self, G: nx.DiGraph, U: nx.Graph, nodes: list, **kwargs) -> Dict[str, Dict[str, Any]]:
        result = {}
        
        for node in nodes:
            neighbors = list(U.neighbors(node))
            if not neighbors:
                result[node] = {"resource_allocation_sum": 0.0, "resource_allocation_avg": 0.0}
                continue
            
            scores = []
            node_neighbors = set(U.neighbors(node))
            
            for neighbor in neighbors:
                neighbor_neighbors = set(U.neighbors(neighbor))
                common = node_neighbors & neighbor_neighbors
                
                ra_score = 0.0
                for common_neighbor in common:
                    degree = U.degree(common_neighbor)
                    if degree > 0:
                        ra_score += 1.0 / degree
                
                scores.append(ra_score)
            
            if scores:
                result[node] = {
                    "resource_allocation_sum": float(np.sum(scores)),
                    "resource_allocation_avg": float(np.mean(scores)),
                }
            else:
                result[node] = {"resource_allocation_sum": 0.0, "resource_allocation_avg": 0.0}
        
        return result