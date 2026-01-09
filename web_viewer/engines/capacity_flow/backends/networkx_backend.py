"""
NetworkX Flow Backend

Max flow computation using NetworkX algorithms.

Location: web_viewer/engines/capacity_flow/backends/networkx_backend.py
"""
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import networkx as nx

from .base import BaseFlowBackend

logger = logging.getLogger(__name__)


class NetworkXBackend(BaseFlowBackend):
    """
    NetworkX-based max flow backend.
    
    Supports multiple algorithms:
    - edmonds_karp (supports cutoff)
    - shortest_augmenting_path (supports cutoff)
    - preflow_push (fastest, no cutoff support)
    - boykov_kolmogorov (no cutoff support)
    - dinitz (no cutoff support)
    """
    
    name = "networkx"
    DEFAULT_ALGORITHM = "edmonds_karp"
    
    ALGORITHMS = {
        "edmonds_karp": nx.algorithms.flow.edmonds_karp,
        "shortest_augmenting_path": nx.algorithms.flow.shortest_augmenting_path,
        "preflow_push": nx.algorithms.flow.preflow_push,
        "boykov_kolmogorov": nx.algorithms.flow.boykov_kolmogorov,
        "dinitz": nx.algorithms.flow.dinitz,
    }
    
    # Algorithms that support cutoff parameter
    CUTOFF_SUPPORTED = {"edmonds_karp", "shortest_augmenting_path"}
    
    def compute_max_flow(
        self,
        edges: List[Tuple[str, str, int]],
        source: str,
        sink: str,
        algorithm: Optional[str] = None,
        cutoff: Optional[int] = None
    ) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """Compute max flow using NetworkX."""
        
        algorithm = algorithm or self.DEFAULT_ALGORITHM
        
        if algorithm not in self.ALGORITHMS:
            logger.warning(f"Unknown algorithm {algorithm}, using {self.DEFAULT_ALGORITHM}")
            algorithm = self.DEFAULT_ALGORITHM
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        for src, tgt, cap in edges:
            if G.has_edge(src, tgt):
                G[src][tgt]['capacity'] += cap
            else:
                G.add_edge(src, tgt, capacity=cap)
        
        if source not in G:
            logger.error(f"Source {source} not in graph")
            return 0, {}
        
        if sink not in G:
            logger.error(f"Sink {sink} not in graph")
            return 0, {}
        
        flow_func = self.ALGORITHMS[algorithm]
        
        try:
            # Only pass cutoff to algorithms that support it
            if cutoff is not None and algorithm in self.CUTOFF_SUPPORTED:
                logger.info(f"Using {algorithm} with cutoff={cutoff}")
                R = flow_func(G, source, sink, cutoff=cutoff)
            else:
                if cutoff is not None and algorithm not in self.CUTOFF_SUPPORTED:
                    logger.info(f"Algorithm {algorithm} doesn't support cutoff, ignoring cutoff={cutoff}")
                R = flow_func(G, source, sink)
            
            flow_value = R.graph.get('flow_value', 0)
            
            # Convert flow to integers and filter zeros
            flow_dict: Dict[str, Dict[str, int]] = defaultdict(dict)
            
            for u in R.nodes():
                for v, data in R[u].items():
                    flow = data.get('flow', 0)
                    if flow > 0:
                        flow_dict[u][v] = int(flow)
            
            return int(flow_value), dict(flow_dict)
            
        except nx.NetworkXError as e:
            logger.error(f"NetworkX error: {e}")
            return 0, {}
    
    @classmethod
    def is_available(cls) -> bool:
        """NetworkX is always available."""
        return True
    
    @classmethod
    def supported_algorithms(cls) -> List[str]:
        """Return supported algorithms."""
        return list(cls.ALGORITHMS.keys())
    
    @classmethod
    def get_algorithm_info(cls) -> List[Dict[str, Any]]:
        """Return detailed info about each algorithm."""
        return [
            {
                "id": "networkx:edmonds_karp",
                "name": "edmonds_karp",
                "label": "Edmonds-Karp (NetworkX)",
                "backend": "networkx",
                "algorithm": "edmonds_karp",
                "supports_cutoff": True,
                "description": "BFS-based, good for sparse graphs"
            },
            {
                "id": "networkx:shortest_augmenting_path",
                "name": "shortest_augmenting_path",
                "label": "Shortest Augmenting Path (NetworkX)",
                "backend": "networkx",
                "algorithm": "shortest_augmenting_path",
                "supports_cutoff": True,
                "description": "Similar to Edmonds-Karp"
            },
            {
                "id": "networkx:preflow_push",
                "name": "preflow_push",
                "label": "Preflow Push (NetworkX)",
                "backend": "networkx",
                "algorithm": "preflow_push",
                "supports_cutoff": False,
                "description": "Fastest for dense graphs"
            },
            {
                "id": "networkx:dinitz",
                "name": "dinitz",
                "label": "Dinitz (NetworkX)",
                "backend": "networkx",
                "algorithm": "dinitz",
                "supports_cutoff": False,
                "description": "Good general performance"
            },
            {
                "id": "networkx:boykov_kolmogorov",
                "name": "boykov_kolmogorov",
                "label": "Boykov-Kolmogorov (NetworkX)",
                "backend": "networkx",
                "algorithm": "boykov_kolmogorov",
                "supports_cutoff": False,
                "description": "Good for image segmentation"
            },
        ]