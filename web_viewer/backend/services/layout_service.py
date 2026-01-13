"""
Layout Service

Handles graph layout computation with multiple backends:
1. Cache lookup
2. Cytoscape Desktop (py4cytoscape)
3. igraph (python-igraph)
4. ForceAtlas2 (fa2)
5. External layout service (Node.js)
6. Local spring layout (NumPy)
7. Circular layout (fallback)
"""

import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import networkx as nx
import requests

from ..config import settings, HAS_CYTOSCAPE_DESKTOP

if HAS_CYTOSCAPE_DESKTOP:
    import py4cytoscape as p4c

from .layout_backends import (
    IGraphLayoutBackend,
    FA2LayoutBackend,
    get_backend_info,
    HAS_IGRAPH,
    HAS_FA2
)


class LocalSpringLayout:
    """
    Local spring-based layout algorithm using NumPy.
    
    Implements force-directed layout with:
    - Spring forces between connected nodes
    - Repulsion forces between all nodes
    - Velocity damping for stability
    """
    
    def __init__(
        self,
        spring_strength: float = None,
        spring_length: float = None,
        repulsion_strength: float = None,
        damping: float = None,
        max_velocity: float = None,
        convergence_threshold: float = None,
        max_iterations: int = None
    ):
        """Initialize with configurable parameters."""
        self.spring_strength = spring_strength or settings.SPRING_STRENGTH
        self.spring_length = spring_length or settings.SPRING_LENGTH
        self.repulsion_strength = repulsion_strength or settings.REPULSION_STRENGTH
        self.damping = damping or settings.DAMPING
        self.max_velocity = max_velocity or settings.MAX_VELOCITY
        self.convergence_threshold = convergence_threshold or settings.CONVERGENCE_THRESHOLD
        self.max_iterations = max_iterations or settings.MAX_ITERATIONS
    
    def compute_layout(
        self,
        G: nx.Graph,
        fixed_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        new_nodes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute spring layout for graph.
        
        Args:
            G: NetworkX graph
            fixed_positions: Pre-existing positions to anchor
            new_nodes: List of new nodes to position (others are fixed)
            
        Returns:
            Positions dictionary {node_id: {x, y}}
        """
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Initialize positions
        positions = np.zeros((n, 2))
        fixed_mask = np.zeros(n, dtype=bool)
        
        if fixed_positions:
            for node, (x, y) in fixed_positions.items():
                if node in node_to_idx:
                    idx = node_to_idx[node]
                    positions[idx] = [x, y]
                    if new_nodes is None or node not in new_nodes:
                        fixed_mask[idx] = True
        
        # Random initialization for unfixed nodes
        unfixed_indices = np.where(~fixed_mask)[0]
        if len(unfixed_indices) > 0:
            if fixed_positions:
                fixed_indices = np.where(fixed_mask)[0]
                if len(fixed_indices) > 0:
                    centroid = positions[fixed_indices].mean(axis=0)
                    spread = max(100, np.std(positions[fixed_indices]) * 2)
                    positions[unfixed_indices] = centroid + np.random.randn(len(unfixed_indices), 2) * spread
                else:
                    positions[unfixed_indices] = np.random.randn(len(unfixed_indices), 2) * 500
            else:
                positions[unfixed_indices] = np.random.randn(len(unfixed_indices), 2) * 500
        
        # Build edge list
        edges = []
        for u, v in G.edges():
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
        edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
        
        velocity = np.zeros((n, 2))
        
        for iteration in range(self.max_iterations):
            forces = np.zeros((n, 2))
            
            # Repulsion forces
            if n > 1:
                for i in range(n):
                    if fixed_mask[i]:
                        continue
                    
                    diff = positions[i] - positions
                    dist_sq = np.sum(diff ** 2, axis=1)
                    dist_sq[i] = 1
                    dist_sq = np.maximum(dist_sq, 1)
                    
                    force_mag = self.repulsion_strength / dist_sq
                    dist = np.sqrt(dist_sq)
                    direction = diff / dist[:, np.newaxis]
                    direction[i] = 0
                    
                    forces[i] = np.sum(force_mag[:, np.newaxis] * direction, axis=0)
            
            # Spring forces
            if len(edges) > 0:
                for u_idx, v_idx in edges:
                    diff = positions[v_idx] - positions[u_idx]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        displacement = dist - self.spring_length
                        force_mag = self.spring_strength * displacement
                        direction = diff / dist
                        force = force_mag * direction
                        
                        if not fixed_mask[u_idx]:
                            forces[u_idx] += force
                        if not fixed_mask[v_idx]:
                            forces[v_idx] -= force
            
            velocity = velocity * self.damping + forces
            
            speed = np.linalg.norm(velocity, axis=1, keepdims=True)
            speed = np.maximum(speed, 1e-10)
            velocity = np.where(
                speed > self.max_velocity,
                velocity * self.max_velocity / speed,
                velocity
            )
            
            positions[~fixed_mask] += velocity[~fixed_mask]
            
            max_movement = np.max(np.linalg.norm(velocity[~fixed_mask], axis=1)) if np.any(~fixed_mask) else 0
            if max_movement < self.convergence_threshold:
                print(f"[LAYOUT] Converged at iteration {iteration}")
                break
        
        result = {}
        for node, idx in node_to_idx.items():
            result[str(node)] = {
                'x': float(positions[idx, 0]),
                'y': float(positions[idx, 1])
            }
        
        return result


class LayoutService:
    """
    Service for computing graph layouts with multiple backends.
    
    Backend priority (configurable via LAYOUT_BACKEND_PRIORITY):
    1. Cache lookup
    2. Cytoscape Desktop (if available and graph is small enough)
    3. igraph (if available)
    4. ForceAtlas2 (if available)
    5. External layout service
    6. Local spring layout
    7. Circular layout (fallback)
    """
    
    def __init__(self):
        """Initialize layout service with all available backends."""
        self.local_spring = LocalSpringLayout()
        self.cytoscape_available = self._check_cytoscape()
        
        # Initialize new backends
        self.igraph_backend = IGraphLayoutBackend()
        self.fa2_backend = FA2LayoutBackend()
        
        # Log available backends
        print(f"[LAYOUT] Available backends:")
        print(f"  Cytoscape Desktop: {'Y' if self.cytoscape_available else 'N'}")
        print(f"  igraph: {'Y' if self.igraph_backend.is_available else 'N'}")
        print(f"  ForceAtlas2: {'Y' if self.fa2_backend.is_available else 'N'}")
        print(f"  Local Spring: Y")
        print(f"  Circular: Y")
    
    def _check_cytoscape(self) -> bool:
        """Check if Cytoscape Desktop is available."""
        if not HAS_CYTOSCAPE_DESKTOP:
            return False
        try:
            p4c.cytoscape_ping()
            print("[LAYOUT] Cytoscape Desktop available")
            return True
        except Exception:
            return False
    
    def get_available_backends(self) -> List[Dict]:
        """Get list of available backends with their info."""
        backends = []
        
        backends.append({
            "id": "cytoscape_desktop",
            "name": "Cytoscape Desktop",
            "available": self.cytoscape_available,
            "description": "Professional layouts via CyREST API",
            "algorithms": ["force-directed-cl"],
            "requires": "Cytoscape Desktop running"
        })
        
        backends.extend(get_backend_info())
        
        backends.append({
            "id": "layout_service",
            "name": "External Layout Service",
            "available": True,
            "description": "Node.js Cytoscape.js headless",
            "algorithms": ["cose"],
            "requires": "Node.js server"
        })
        
        backends.append({
            "id": "local_spring",
            "name": "Local Spring (NumPy)",
            "available": True,
            "description": "Built-in force-directed layout",
            "algorithms": ["spring"],
            "requires": "None"
        })
        
        backends.append({
            "id": "circular",
            "name": "Circular",
            "available": True,
            "description": "Simple circular layout (fallback)",
            "algorithms": ["circular"],
            "requires": "None"
        })
        
        return backends
    
    def compute_layout(
        self,
        G: nx.Graph,
        graph_id: str,
        cached_layout: Optional[Dict[str, Dict[str, float]]] = None,
        use_cytoscape: bool = True,
        preferred_backend: Optional[str] = None,
        algorithm: Optional[str] = None,
        **kwargs
    ) -> Tuple[Dict[str, Dict[str, float]], str, float]:
        """
        Compute layout for graph using best available backend.
        
        Args:
            G: NetworkX graph
            graph_id: Graph identifier
            cached_layout: Pre-computed layout to use if available
            use_cytoscape: Whether to try Cytoscape Desktop
            preferred_backend: Force specific backend
            algorithm: Algorithm for backends that support multiple
            **kwargs: Backend-specific parameters
            
        Returns:
            Tuple of (positions, algorithm_name, computation_time)
        """
        start_time = time.time()
        
        # Use cached layout if available
        if cached_layout:
            positions = {
                node: cached_layout[node]
                for node in G.nodes()
                if node in cached_layout
            }
            
            if len(positions) == len(G.nodes()):
                return positions, "cached", time.time() - start_time
            elif len(positions) > 0:
                print(f"[LAYOUT] Partial cache hit: {len(positions)}/{len(G.nodes())} nodes")
                new_nodes = [n for n in G.nodes() if n not in positions]
                positions = self.compute_incremental_layout(G, positions, new_nodes)
                return positions, "incremental", time.time() - start_time
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # If preferred backend specified, try it first
        if preferred_backend:
            positions = self._try_backend(
                preferred_backend, G, graph_id, n_edges, use_cytoscape, algorithm, **kwargs
            )
            if positions:
                return positions, f"{preferred_backend}", time.time() - start_time
        
        # Follow configured priority order
        for backend in settings.LAYOUT_BACKEND_PRIORITY:
            if backend == "cached":
                continue  # Already handled
            
            positions = self._try_backend(
                backend, G, graph_id, n_edges, use_cytoscape, algorithm, **kwargs
            )
            if positions:
                return positions, backend, time.time() - start_time
        
        # Final fallback
        print(f"[LAYOUT] All backends failed, using circular for {n_nodes} nodes")
        positions = self.compute_circular_layout(G)
        return positions, "circular", time.time() - start_time
    
    def _try_backend(
        self,
        backend: str,
        G: nx.Graph,
        graph_id: str,
        n_edges: int,
        use_cytoscape: bool,
        algorithm: Optional[str] = None,
        initial_positions: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Try a specific layout backend.
        
        Args:
            backend: Backend name
            G: NetworkX graph  
            graph_id: Graph identifier
            n_edges: Number of edges
            use_cytoscape: Whether to try Cytoscape Desktop
            algorithm: Algorithm for backends that support multiple
            initial_positions: Starting positions for warm start
            **kwargs: Backend-specific parameters
        """
        n_nodes = G.number_of_nodes()
        
        if backend == "cytoscape_desktop":
            if use_cytoscape and self.cytoscape_available and n_edges <= settings.MAX_EDGES_FOR_CYTOSCAPE_DESKTOP:
                try:
                    return self.compute_layout_via_cytoscape_desktop(G, graph_id)
                except Exception as e:
                    print(f"[LAYOUT] Cytoscape Desktop failed: {e}")
        
        elif backend == "igraph":
            if self.igraph_backend.is_available:
                try:
                    algo = algorithm or settings.IGRAPH_DEFAULT_ALGORITHM
                    positions = self.igraph_backend.compute_layout(
                        G,
                        algorithm=algo,
                        scale=settings.IGRAPH_SCALE,
                        initial_positions=initial_positions,
                        **kwargs
                    )
                    if positions:
                        return positions
                except Exception as e:
                    print(f"[LAYOUT] igraph failed: {e}")
        
        elif backend == "fa2":
            if self.fa2_backend.is_available:
                try:
                    positions = self.fa2_backend.compute_layout(
                        G,
                        iterations=kwargs.get('iterations', settings.FA2_ITERATIONS),
                        barnes_hut_optimize=kwargs.get('barnes_hut_optimize', settings.FA2_BARNES_HUT_OPTIMIZE),
                        barnes_hut_theta=kwargs.get('barnes_hut_theta', settings.FA2_BARNES_HUT_THETA),
                        scaling_ratio=kwargs.get('scaling_ratio', settings.FA2_SCALING_RATIO),
                        gravity=kwargs.get('gravity', settings.FA2_GRAVITY),
                        scale=kwargs.get('scale', settings.FA2_SCALE),
                        initial_positions=initial_positions,
                    )
                    if positions:
                        return positions
                except Exception as e:
                    print(f"[LAYOUT] ForceAtlas2 failed: {e}")
        
        elif backend == "layout_service":
            try:
                positions = self.compute_layout_via_service(G)
                if positions:
                    return positions
            except Exception as e:
                print(f"[LAYOUT] External service failed: {e}")
        
        elif backend == "local_spring":
            if n_nodes <= settings.MAX_NODES_FOR_LOCAL_SPRING:
                print(f"[LAYOUT] Using local spring for {n_nodes} nodes")
                return self.local_spring.compute_layout(G)
        
        elif backend == "circular":
            return self.compute_circular_layout(G)
        
        return None
    
    def compute_layout_via_cytoscape_desktop(
        self, 
        G: nx.Graph, 
        graph_id: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Compute layout using Cytoscape Desktop via CyREST."""
        if not self.cytoscape_available:
            return None
        
        print(f"[LAYOUT] Computing via Cytoscape Desktop ({G.number_of_nodes()} nodes)")
        
        net_suid = None
        try:
            nodes_payload = [{"data": {"id": str(node)}} for node in G.nodes()]
            edges_payload = [
                {"data": {"source": str(src), "target": str(tgt)}} 
                for src, tgt in G.edges()
            ]
            
            title = f"web_viewer_{graph_id}_{int(time.time())}"
            
            print(f"[LAYOUT] Creating network via CyREST...")
            res = p4c.cyrest_post("networks", body={
                "data": {"name": title},
                "elements": {"nodes": nodes_payload, "edges": edges_payload}
            })
            net_suid = res['networkSUID']
            
            try:
                p4c.cyrest_post(f"networks/{net_suid}/views")
                time.sleep(0.2)
            except Exception as e:
                print(f"[LAYOUT] View creation note: {e}")
            
            print(f"[LAYOUT] Applying force-directed layout...")
            p4c.set_layout_properties(
                'force-directed-cl',
                {
                    'numIterations': 400,
                    'numIterationsEdgeRepulsive': 10,
                    'defaultSpringCoefficient': 1e-5,
                    'defaultSpringLength': 30,
                    'defaultNodeMass': 1.0,
                    'isDeterministic': True,
                    'fromScratch': True,
                    'singlePartition': False
                }
            )
            
            p4c.layout_network("force-directed-cl", network=net_suid)
            
            print(f"[LAYOUT] Getting positions from view...")
            views = p4c.get_network_views(net_suid)
            if not views:
                raise RuntimeError("No view found after layout")
            
            view_suid = views[0]
            view_json = p4c.cyrest_get(f"networks/{net_suid}/views/{view_suid}")
            
            positions = {}
            
            if view_json and isinstance(view_json, dict):
                elements = view_json.get('elements', {})
                nodes = elements.get('nodes', [])
                
                for node in nodes:
                    if isinstance(node, dict):
                        node_data = node.get('data', {})
                        node_position = node.get('position', {})
                        node_id = (
                            node_data.get('name') or 
                            node_data.get('shared_name') or 
                            node_data.get('id')
                        )
                        
                        if node_id and 'x' in node_position and 'y' in node_position:
                            positions[node_id] = {
                                'x': float(node_position['x']), 
                                'y': float(node_position['y'])
                            }
            
            try:
                p4c.delete_network(net_suid)
            except Exception:
                pass
            
            print(f"[LAYOUT] Cytoscape Desktop complete: {len(positions)} positions")
            
            if len(positions) == 0:
                raise RuntimeError("No positions retrieved from Cytoscape")
            
            return positions
            
        except Exception as e:
            print(f"[LAYOUT] Cytoscape Desktop error: {e}")
            try:
                if net_suid is not None:
                    p4c.delete_network(net_suid)
            except Exception:
                pass
            return None
    
    def compute_layout_via_service(
        self, 
        G: nx.Graph
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Compute layout via external Node.js service."""
        try:
            elements = []
            
            for node in G.nodes():
                elements.append({
                    'group': 'nodes',
                    'data': {'id': str(node)}
                })
            
            for u, v in G.edges():
                elements.append({
                    'group': 'edges',
                    'data': {
                        'id': f"{u}-{v}",
                        'source': str(u),
                        'target': str(v)
                    }
                })
            
            response = requests.post(
                settings.LAYOUT_SERVICE_URL,
                json={'elements': elements},
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                positions = {}
                for node_pos in data.get('positions', []):
                    positions[node_pos['id']] = {
                        'x': node_pos['x'],
                        'y': node_pos['y']
                    }
                print(f"[LAYOUT] Service complete: {len(positions)} positions")
                return positions
            else:
                print(f"[LAYOUT] Service error: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"[LAYOUT] Service unavailable: {e}")
            return None
    
    def compute_circular_layout(
        self, 
        G: nx.Graph,
        radius: float = 1000
    ) -> Dict[str, Dict[str, float]]:
        """Compute simple circular layout as fallback."""
        nodes = list(G.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            positions[str(node)] = {
                'x': float(radius * np.cos(angle)),
                'y': float(radius * np.sin(angle))
            }
        
        return positions
    
    def compute_incremental_layout(
        self,
        G: nx.Graph,
        existing_positions: Dict[str, Dict[str, float]],
        new_nodes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute positions for new nodes while keeping existing nodes fixed."""
        if not new_nodes:
            return existing_positions
        
        print(f"[LAYOUT] Incremental layout for {len(new_nodes)} new nodes")
        
        fixed_positions = {
            node: (pos['x'], pos['y'])
            for node, pos in existing_positions.items()
        }
        
        positions = self.local_spring.compute_layout(
            G,
            fixed_positions=fixed_positions,
            new_nodes=new_nodes
        )
        
        return positions
    
    def recompute_layout(
        self,
        G: nx.Graph,
        graph_id: str,
        backend: Optional[str] = None,
        algorithm: Optional[str] = None,
        initial_positions: Optional[Dict[str, Dict[str, float]]] = None,
        from_scratch: bool = True,
        **kwargs
    ) -> Tuple[Dict[str, Dict[str, float]], str, str, float]:
        """
        Recompute layout for an existing graph.
        
        Args:
            G: NetworkX graph
            graph_id: Graph identifier
            backend: Specific backend to use
            algorithm: Algorithm for backends that support multiple
            initial_positions: Starting positions for warm start (overrides from_scratch)
            from_scratch: If False and no initial_positions, use existing layout as starting point
            **kwargs: Backend-specific parameters
            
        Returns:
            Tuple of (positions, backend_name, algorithm_name, computation_time)
        """
        start_time = time.time()
        
        # Determine initial positions for warm start
        init_pos = None
        if initial_positions:
            init_pos = initial_positions
        elif not from_scratch:
            # Try to load existing layout for warm start
            from .cache_service import CacheService
            cache = CacheService()
            init_pos = cache.get_cached_layout(graph_id)
            if init_pos:
                print(f"[LAYOUT] Using existing layout as starting point ({len(init_pos)} positions)")
        
        n_edges = G.number_of_edges()
        
        # If specific backend requested
        if backend:
            positions = self._try_backend(
                backend, G, graph_id, n_edges, 
                use_cytoscape=True,
                algorithm=algorithm,
                initial_positions=init_pos,
                **kwargs
            )
            if positions:
                algo_name = algorithm or (backend if backend == 'cytoscape_desktop' else 'auto')
                return positions, backend, algo_name, time.time() - start_time
        
        # Try backends in priority order
        for backend_name in settings.LAYOUT_BACKEND_PRIORITY:
            if backend_name == "cached":
                continue  # Skip cache for recompute
            
            positions = self._try_backend(
                backend_name, G, graph_id, n_edges,
                use_cytoscape=True,
                algorithm=algorithm,
                initial_positions=init_pos,
                **kwargs
            )
            if positions:
                algo_name = algorithm or backend_name
                return positions, backend_name, algo_name, time.time() - start_time
        
        # Fallback to circular
        positions = self.compute_circular_layout(G)
        return positions, "circular", "circular", time.time() - start_time