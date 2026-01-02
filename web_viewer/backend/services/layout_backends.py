"""
Layout Backend Implementations

Modular layout backends for graph positioning:
- IGraphLayoutBackend: Fast C-based layouts via python-igraph
- FA2LayoutBackend: Gephi's ForceAtlas2 algorithm
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple
import time
import networkx as nx
import numpy as np


# =============================================================================
# FEATURE FLAGS
# =============================================================================

try:
    import igraph as ig
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False
    ig = None

try:
    # Try fa2_modified first (maintained fork, works with Python 3.9+)
    from fa2_modified import ForceAtlas2
    HAS_FA2 = True
    FA2_PACKAGE = 'fa2_modified'
except ImportError:
    try:
        # Fallback to original fa2 (Python <3.8 only)
        from fa2 import ForceAtlas2
        HAS_FA2 = True
        FA2_PACKAGE = 'fa2'
    except ImportError:
        HAS_FA2 = False
        ForceAtlas2 = None
        FA2_PACKAGE = None


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class LayoutBackend(ABC):
    """Abstract base class for layout backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier name."""
        pass
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass
    
    @abstractmethod
    def compute_layout(
        self,
        G: nx.Graph,
        **kwargs
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute layout positions for all nodes.
        
        Args:
            G: NetworkX graph
            **kwargs: Backend-specific parameters
            
        Returns:
            Dict mapping node_id -> {'x': float, 'y': float}
            None if computation failed
        """
        pass
    
    def get_algorithms(self) -> List[str]:
        """Return list of supported algorithms."""
        return []


# =============================================================================
# IGRAPH BACKEND
# =============================================================================

class IGraphLayoutBackend(LayoutBackend):
    """
    Layout backend using python-igraph.
    
    Algorithms:
    - drl: Distributed Recursive Layout (large graphs, 10K-1M nodes)
    - fr: Fruchterman-Reingold (medium graphs, <5K nodes)
    - kk: Kamada-Kawai (small graphs, <1K nodes, best quality)
    - lgl: Large Graph Layout (very large graphs)
    - graphopt: General purpose
    """
    
    ALGORITHMS = ['auto', 'drl', 'fr', 'kk', 'lgl', 'graphopt', 'circle', 'grid', 'random']
    
    # Thresholds for auto algorithm selection
    THRESHOLDS = [
        (500, 'kk'),
        (3000, 'fr'),
        (50000, 'drl'),
        (float('inf'), 'lgl')
    ]
    
    @property
    def name(self) -> str:
        return "igraph"
    
    @property
    def is_available(self) -> bool:
        return HAS_IGRAPH
    
    def get_algorithms(self) -> List[str]:
        return self.ALGORITHMS
    
    def _nx_to_igraph(self, G: nx.Graph) -> Tuple:
        """Convert NetworkX graph to igraph."""
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        ig_graph = ig.Graph(directed=G.is_directed())
        ig_graph.add_vertices(len(nodes))
        
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges() 
                 if u in node_to_idx and v in node_to_idx]
        if edges:
            ig_graph.add_edges(edges)
        
        return ig_graph, node_to_idx, idx_to_node
    
    def _select_algorithm(self, n_nodes: int, algorithm: str) -> str:
        """Select algorithm based on graph size if auto."""
        if algorithm and algorithm != 'auto':
            return algorithm
        for threshold, algo in self.THRESHOLDS:
            if n_nodes < threshold:
                return algo
        return 'drl'
    
    def compute_layout(
        self,
        G: nx.Graph,
        algorithm: str = 'auto',
        scale: float = 1000.0,
        **kwargs
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Compute layout using igraph."""
        if not self.is_available:
            return None
        
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            return {}
        
        try:
            start = time.time()
            ig_graph, node_to_idx, idx_to_node = self._nx_to_igraph(G)
            
            algo = self._select_algorithm(n_nodes, algorithm)
            print(f"[LAYOUT:igraph] {algo} for {n_nodes} nodes, {G.number_of_edges()} edges")
            
            # Compute layout based on algorithm
            if algo == 'drl':
                layout = ig_graph.layout_drl()
            elif algo == 'fr':
                niter = kwargs.get('niter', 500)
                layout = ig_graph.layout_fruchterman_reingold(niter=niter)
            elif algo == 'kk':
                maxiter = kwargs.get('maxiter', 1000)
                layout = ig_graph.layout_kamada_kawai(maxiter=maxiter)
            elif algo == 'lgl':
                layout = ig_graph.layout_lgl()
            elif algo == 'graphopt':
                niter = kwargs.get('niter', 500)
                layout = ig_graph.layout_graphopt(niter=niter)
            elif algo == 'circle':
                layout = ig_graph.layout_circle()
            elif algo == 'grid':
                layout = ig_graph.layout_grid()
            elif algo == 'random':
                layout = ig_graph.layout_random()
            else:
                layout = ig_graph.layout_auto()
            
            # Scale positions
            layout.scale(scale)
            coords = layout.coords
            
            # Convert to output format
            positions = {}
            for idx, (x, y) in enumerate(coords):
                node_id = str(idx_to_node[idx])
                positions[node_id] = {'x': float(x), 'y': float(y)}
            
            elapsed = time.time() - start
            print(f"[LAYOUT:igraph] Complete: {len(positions)} positions in {elapsed:.2f}s")
            return positions
            
        except Exception as e:
            print(f"[LAYOUT:igraph] Error: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# FORCEATLAS2 BACKEND
# =============================================================================

class FA2LayoutBackend(LayoutBackend):
    """
    Layout backend using ForceAtlas2 algorithm.
    
    Gephi's ForceAtlas2 with Barnes-Hut optimization for O(n log n) complexity.
    Excellent for social/trust networks with hub structures.
    """
    
    @property
    def name(self) -> str:
        return "fa2"
    
    @property
    def is_available(self) -> bool:
        return HAS_FA2
    
    def get_algorithms(self) -> List[str]:
        return ['forceatlas2']
    
    def compute_layout(
        self,
        G: nx.Graph,
        iterations: int = 1000,
        barnes_hut_optimize: bool = True,
        barnes_hut_theta: float = 1.2,
        scaling_ratio: float = 2.0,
        gravity: float = 1.0,
        outbound_attraction_distribution: bool = True,
        scale: float = 1.0,
        **kwargs
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Compute layout using ForceAtlas2."""
        if not self.is_available:
            return None
        
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            return {}
        
        try:
            start = time.time()
            print(f"[LAYOUT:fa2] ForceAtlas2 for {n_nodes} nodes, {iterations} iterations")
            
            # Adjust iterations for large graphs
            if n_nodes > 10000 and iterations > 500:
                iterations = 500
                print(f"[LAYOUT:fa2] Reduced iterations to {iterations} for large graph")
            
            forceatlas2 = ForceAtlas2(
                outboundAttractionDistribution=outbound_attraction_distribution,
                linLogMode=False,
                adjustSizes=False,
                edgeWeightInfluence=1.0,
                jitterTolerance=1.0,
                barnesHutOptimize=barnes_hut_optimize,
                barnesHutTheta=barnes_hut_theta,
                multiThreaded=False,
                scalingRatio=scaling_ratio,
                strongGravityMode=False,
                gravity=gravity,
                verbose=False
            )
            
            # ForceAtlas2 works with undirected graphs
            if G.is_directed():
                G_undirected = G.to_undirected()
            else:
                G_undirected = G
            
            positions_raw = forceatlas2.forceatlas2_networkx_layout(
                G_undirected,
                pos=None,
                iterations=iterations
            )
            
            # Convert to output format with scaling
            positions = {}
            for node, (x, y) in positions_raw.items():
                positions[str(node)] = {
                    'x': float(x) * scale,
                    'y': float(y) * scale
                }
            
            elapsed = time.time() - start
            print(f"[LAYOUT:fa2] Complete: {len(positions)} positions in {elapsed:.2f}s")
            return positions
            
        except Exception as e:
            print(f"[LAYOUT:fa2] Error: {e}")
            import traceback
            traceback.print_exc()
            return None


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

def get_available_backends() -> Dict[str, LayoutBackend]:
    """Get all available layout backends."""
    backends = {}
    
    igraph_backend = IGraphLayoutBackend()
    if igraph_backend.is_available:
        backends['igraph'] = igraph_backend
    
    fa2_backend = FA2LayoutBackend()
    if fa2_backend.is_available:
        backends['fa2'] = fa2_backend
    
    return backends


def get_backend_info() -> List[Dict]:
    """Get information about all backends."""
    info = []
    
    info.append({
        "id": "igraph",
        "name": "igraph",
        "available": HAS_IGRAPH,
        "description": "Fast C-based graph layouts (DrL, FR, KK)",
        "algorithms": IGraphLayoutBackend.ALGORITHMS if HAS_IGRAPH else [],
        "recommended_max_nodes": "100K+",
        "install": "pip install igraph"
    })
    
    info.append({
        "id": "fa2",
        "name": "ForceAtlas2",
        "available": HAS_FA2,
        "description": "Gephi's ForceAtlas2 with Barnes-Hut optimization",
        "algorithms": ["forceatlas2"],
        "recommended_max_nodes": "50K",
        "install": "pip install fa2_modified",
        "package": FA2_PACKAGE if HAS_FA2 else None
    })
    
    return info