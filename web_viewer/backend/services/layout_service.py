"""
Layout Service

Handles graph layout computation with multiple backends:
1. Cache lookup
2. Cytoscape Desktop (py4cytoscape)
3. External layout service (Node.js)
4. Local spring layout (NumPy)
5. Circular layout (fallback)
"""

import time
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import networkx as nx
import requests

from ..config import settings, HAS_CYTOSCAPE_DESKTOP

if HAS_CYTOSCAPE_DESKTOP:
    import py4cytoscape as p4c


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
                # Place near centroid of fixed nodes
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
        
        # Velocity
        velocity = np.zeros((n, 2))
        
        # Iterate
        for iteration in range(self.max_iterations):
            forces = np.zeros((n, 2))
            
            # Repulsion forces (all pairs) - use batched computation for efficiency
            if n > 1:
                for i in range(n):
                    if fixed_mask[i]:
                        continue
                    
                    # Vector from all other nodes to this node
                    diff = positions[i] - positions
                    dist_sq = np.sum(diff ** 2, axis=1)
                    dist_sq[i] = 1  # Avoid self
                    dist_sq = np.maximum(dist_sq, 1)  # Minimum distance
                    
                    # Repulsion force magnitude
                    force_mag = self.repulsion_strength / dist_sq
                    
                    # Direction
                    dist = np.sqrt(dist_sq)
                    direction = diff / dist[:, np.newaxis]
                    direction[i] = 0
                    
                    # Sum forces
                    forces[i] = np.sum(force_mag[:, np.newaxis] * direction, axis=0)
            
            # Spring forces (connected nodes)
            if len(edges) > 0:
                for u_idx, v_idx in edges:
                    diff = positions[v_idx] - positions[u_idx]
                    dist = np.linalg.norm(diff)
                    
                    if dist > 0:
                        # Spring force
                        displacement = dist - self.spring_length
                        force_mag = self.spring_strength * displacement
                        direction = diff / dist
                        
                        force = force_mag * direction
                        
                        if not fixed_mask[u_idx]:
                            forces[u_idx] += force
                        if not fixed_mask[v_idx]:
                            forces[v_idx] -= force
            
            # Update velocity and position
            velocity = velocity * self.damping + forces
            
            # Limit velocity
            speed = np.linalg.norm(velocity, axis=1, keepdims=True)
            speed = np.maximum(speed, 1e-10)
            velocity = np.where(
                speed > self.max_velocity,
                velocity * self.max_velocity / speed,
                velocity
            )
            
            # Update positions for unfixed nodes
            positions[~fixed_mask] += velocity[~fixed_mask]
            
            # Check convergence
            max_movement = np.max(np.linalg.norm(velocity[~fixed_mask], axis=1)) if np.any(~fixed_mask) else 0
            if max_movement < self.convergence_threshold:
                print(f"[LAYOUT] Converged at iteration {iteration}")
                break
        
        # Build result
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
    
    Backend priority:
    1. Cache lookup
    2. Cytoscape Desktop (if available and graph is small enough)
    3. External layout service
    4. Local spring layout
    5. Circular layout (fallback)
    """
    
    def __init__(self):
        """Initialize layout service."""
        self.local_spring = LocalSpringLayout()
        self.cytoscape_available = self._check_cytoscape()
    
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
    
    def compute_layout(
        self,
        G: nx.Graph,
        graph_id: str,
        cached_layout: Optional[Dict[str, Dict[str, float]]] = None,
        use_cytoscape: bool = True
    ) -> Tuple[Dict[str, Dict[str, float]], str, float]:
        """
        Compute layout for graph using best available backend.
        
        Args:
            G: NetworkX graph
            graph_id: Graph identifier
            cached_layout: Pre-computed layout to use if available
            use_cytoscape: Whether to try Cytoscape Desktop
            
        Returns:
            Tuple of (positions, algorithm_name, computation_time)
        """
        start_time = time.time()
        
        # Use cached layout if available
        if cached_layout:
            # Filter to only include nodes in current graph
            positions = {
                node: cached_layout[node]
                for node in G.nodes()
                if node in cached_layout
            }
            
            if len(positions) == len(G.nodes()):
                return positions, "cached", time.time() - start_time
            elif len(positions) > 0:
                # Partial cache - compute incremental layout
                print(f"[LAYOUT] Partial cache hit: {len(positions)}/{len(G.nodes())} nodes")
                new_nodes = [n for n in G.nodes() if n not in positions]
                positions = self.compute_incremental_layout(G, positions, new_nodes)
                return positions, "incremental", time.time() - start_time
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Try Cytoscape Desktop for small-medium graphs
        if (use_cytoscape and self.cytoscape_available and 
            n_edges <= settings.MAX_EDGES_FOR_CYTOSCAPE_DESKTOP):
            try:
                positions = self.compute_layout_via_cytoscape_desktop(G, graph_id)
                if positions:
                    return positions, "cytoscape_desktop", time.time() - start_time
            except Exception as e:
                print(f"[LAYOUT] Cytoscape Desktop failed: {e}")
        
        # Try external layout service
        try:
            positions = self.compute_layout_via_service(G)
            if positions:
                return positions, "layout_service", time.time() - start_time
        except Exception as e:
            print(f"[LAYOUT] External service failed: {e}")
        
        # Fall back to local spring layout for smaller graphs
        if n_nodes <= 10000:
            print(f"[LAYOUT] Using local spring layout for {n_nodes} nodes")
            positions = self.local_spring.compute_layout(G)
            return positions, "local_spring", time.time() - start_time
        
        # Final fallback: circular layout
        print(f"[LAYOUT] Using circular layout for {n_nodes} nodes")
        positions = self.compute_circular_layout(G)
        return positions, "circular", time.time() - start_time
    
    def compute_layout_via_cytoscape_desktop(
        self, 
        G: nx.Graph, 
        graph_id: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute layout using Cytoscape Desktop.
        
        Args:
            G: NetworkX graph
            graph_id: Graph identifier for naming
            
        Returns:
            Positions dictionary or None if failed
        """
        if not self.cytoscape_available:
            return None
        
        print(f"[LAYOUT] Computing via Cytoscape Desktop ({G.number_of_nodes()} nodes)")
        
        try:
            # Create network in Cytoscape
            network_suid = p4c.create_network_from_networkx(
                G, 
                title=graph_id,
                collection="GraphAnalyzer"
            )
            
            # Apply force-directed layout
            p4c.layout_network(
                'force-directed-cl',
                network=network_suid,
                parameters={
                    'numIterations': 400,
                    'defaultSpringLength': 100,
                    'defaultSpringCoefficient': 0.0001
                }
            )
            
            # Get positions
            node_table = p4c.get_node_table(network=network_suid)
            
            positions = {}
            for _, row in node_table.iterrows():
                node_name = str(row.get('name', row.get('SUID', '')))
                x = row.get('x', row.get('X_LOCATION', 0))
                y = row.get('y', row.get('Y_LOCATION', 0))
                positions[node_name] = {'x': float(x), 'y': float(y)}
            
            # Clean up
            p4c.delete_network(network=network_suid)
            
            print(f"[LAYOUT] Cytoscape Desktop complete: {len(positions)} positions")
            return positions
            
        except Exception as e:
            print(f"[LAYOUT] Cytoscape Desktop error: {e}")
            return None
    
    def compute_layout_via_service(
        self, 
        G: nx.Graph
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute layout via external Node.js service.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Positions dictionary or None if failed
        """
        try:
            # Prepare elements for Cytoscape.js
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
            
            # Send to layout service
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
        """
        Compute simple circular layout as fallback.
        
        Args:
            G: NetworkX graph
            radius: Circle radius
            
        Returns:
            Positions dictionary
        """
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
        """
        Compute positions for new nodes while keeping existing nodes fixed.
        
        Args:
            G: NetworkX graph
            existing_positions: Positions of existing nodes
            new_nodes: List of new node IDs to position
            
        Returns:
            Complete positions dictionary
        """
        if not new_nodes:
            return existing_positions
        
        print(f"[LAYOUT] Incremental layout for {len(new_nodes)} new nodes")
        
        # Convert existing positions to tuple format
        fixed_positions = {
            node: (pos['x'], pos['y'])
            for node, pos in existing_positions.items()
        }
        
        # Compute layout with fixed positions
        positions = self.local_spring.compute_layout(
            G,
            fixed_positions=fixed_positions,
            new_nodes=new_nodes
        )
        
        return positions