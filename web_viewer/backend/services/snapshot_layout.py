"""
Snapshot Layout Service

Handles layout derivation and constrained spring layout algorithm for
positioning unknown nodes in historical snapshots.
"""

import math
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

from ..config import settings


class SnapshotLayout:
    """
    Service for deriving layouts for historical snapshots.
    
    Uses a constrained spring layout algorithm where:
    - Known nodes (in master layout) keep their positions
    - Unknown nodes (not in master) are positioned via spring algorithm
    """
    
    def __init__(
        self,
        iterations: Optional[int] = None,
        k: Optional[float] = None,
        repulsion: Optional[float] = None,
        attraction: Optional[float] = None,
        damping: Optional[float] = None
    ):
        """
        Initialize layout service with spring algorithm parameters.
        
        Args:
            iterations: Number of spring iterations
            k: Optimal distance between nodes
            repulsion: Repulsion force strength
            attraction: Attraction force strength
            damping: Damping factor for velocity
        """
        self.iterations = iterations or settings.SNAPSHOT_SPRING_ITERATIONS
        self.k = k or settings.SNAPSHOT_SPRING_K
        self.repulsion = repulsion or settings.SNAPSHOT_SPRING_REPULSION
        self.attraction = attraction or settings.SNAPSHOT_SPRING_ATTRACTION
        self.damping = damping or settings.SNAPSHOT_SPRING_DAMPING
    
    def derive_layout(
        self,
        snapshot_nodes: Set[str],
        edges: List[Tuple[str, str]],
        master_layout: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Derive layout for a snapshot using master layout as reference.
        
        Algorithm:
        1. Partition nodes into known (in master) and unknown (not in master)
        2. If no unknown nodes, return positions directly from master
        3. If unknown nodes exist, run constrained spring layout
        4. Return combined layout and list of unknown nodes
        
        Args:
            snapshot_nodes: Set of node IDs in the snapshot
            edges: List of (source, target) tuples
            master_layout: Master layout {node_id: {x, y, first_seen}}
            
        Returns:
            Tuple of:
            - layout_dict: {node_id: {x, y}} for all snapshot nodes
            - unknown_nodes: List of nodes that were positioned via spring
        """
        # Partition nodes
        master_nodes = set(master_layout.keys())
        known_nodes = snapshot_nodes & master_nodes
        unknown_nodes = snapshot_nodes - master_nodes
        
        print(f"[LAYOUT] Snapshot nodes: {len(snapshot_nodes)}, "
              f"known: {len(known_nodes)}, unknown: {len(unknown_nodes)}")
        
        # If all nodes are known, return their positions directly
        if not unknown_nodes:
            layout = {
                node: {'x': master_layout[node]['x'], 'y': master_layout[node]['y']}
                for node in snapshot_nodes
            }
            return layout, []
        
        # Build fixed positions from master
        fixed_positions = {
            node: {'x': master_layout[node]['x'], 'y': master_layout[node]['y']}
            for node in known_nodes
        }
        
        # Position unknown nodes via constrained spring
        new_positions = self.position_unknown_nodes(
            edges=edges,
            fixed_positions=fixed_positions,
            free_nodes=unknown_nodes
        )
        
        # Combine all positions
        layout = {}
        for node in snapshot_nodes:
            if node in fixed_positions:
                layout[node] = fixed_positions[node]
            elif node in new_positions:
                layout[node] = new_positions[node]
            else:
                # Fallback: place at origin (shouldn't happen)
                print(f"[LAYOUT] Warning: Node {node} has no position")
                layout[node] = {'x': 0.0, 'y': 0.0}
        
        return layout, list(unknown_nodes)
    
    def position_unknown_nodes(
        self,
        edges: List[Tuple[str, str]],
        fixed_positions: Dict[str, Dict[str, float]],
        free_nodes: Set[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Position unknown nodes using constrained spring layout.
        
        Algorithm:
        1. Build adjacency list from edges
        2. Calculate bounding box from fixed positions
        3. Initialize free node positions near their neighbors
        4. Run spring iterations (only moving free nodes):
           - Repulsion: All pairs involving at least one free node
           - Attraction: All edges involving at least one free node
        5. Return final positions for free nodes only
        
        Args:
            edges: List of (source, target) tuples
            fixed_positions: Positions that cannot move {node: {x, y}}
            free_nodes: Set of nodes that need positions
            
        Returns:
            Positions for free nodes {node: {x, y}}
        """
        if not free_nodes:
            return {}
        
        # Build adjacency list (bidirectional for spring layout)
        adjacency = defaultdict(set)
        for source, target in edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        
        # Calculate bounds from fixed positions
        if fixed_positions:
            xs = [p['x'] for p in fixed_positions.values()]
            ys = [p['y'] for p in fixed_positions.values()]
            bounds = (min(xs), min(ys), max(xs), max(ys))
            center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
            spread = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) or 1000.0
        else:
            center = (0.0, 0.0)
            spread = 1000.0
        
        # Initialize all positions (copy fixed, initialize free)
        positions = dict(fixed_positions)
        
        for node in free_nodes:
            neighbors = adjacency.get(node, set())
            fixed_neighbors = [n for n in neighbors if n in fixed_positions]
            
            if fixed_neighbors:
                # Position near average of fixed neighbors with small random offset
                avg_x = sum(fixed_positions[n]['x'] for n in fixed_neighbors) / len(fixed_neighbors)
                avg_y = sum(fixed_positions[n]['y'] for n in fixed_neighbors) / len(fixed_neighbors)
                offset = spread * 0.05  # 5% of spread
                positions[node] = {
                    'x': avg_x + random.uniform(-offset, offset),
                    'y': avg_y + random.uniform(-offset, offset)
                }
            else:
                # Random position within bounds
                positions[node] = {
                    'x': center[0] + random.uniform(-spread/2, spread/2),
                    'y': center[1] + random.uniform(-spread/2, spread/2)
                }
        
        # Spring iterations
        for iteration in range(self.iterations):
            # Calculate displacements for free nodes only
            displacement = {node: {'x': 0.0, 'y': 0.0} for node in free_nodes}
            
            # Repulsion forces (between all pairs involving free nodes)
            all_nodes = list(positions.keys())
            for i, node1 in enumerate(all_nodes):
                if node1 not in free_nodes:
                    continue
                
                for node2 in all_nodes[i+1:]:
                    dx = positions[node1]['x'] - positions[node2]['x']
                    dy = positions[node1]['y'] - positions[node2]['y']
                    dist = max(math.sqrt(dx*dx + dy*dy), 0.01)
                    
                    # Inverse square repulsion
                    force = self.repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    
                    # Apply to free nodes only
                    if node1 in free_nodes:
                        displacement[node1]['x'] += fx
                        displacement[node1]['y'] += fy
                    if node2 in free_nodes:
                        displacement[node2]['x'] -= fx
                        displacement[node2]['y'] -= fy
            
            # Attraction forces (along edges involving free nodes)
            for source, target in edges:
                # Skip if neither endpoint is free
                if source not in free_nodes and target not in free_nodes:
                    continue
                
                # Skip if either node is not in positions
                if source not in positions or target not in positions:
                    continue
                
                dx = positions[target]['x'] - positions[source]['x']
                dy = positions[target]['y'] - positions[source]['y']
                dist = max(math.sqrt(dx*dx + dy*dy), 0.01)
                
                # Linear spring attraction
                force = self.attraction * dist
                fx = force * dx / dist
                fy = force * dy / dist
                
                if source in free_nodes:
                    displacement[source]['x'] += fx
                    displacement[source]['y'] += fy
                if target in free_nodes:
                    displacement[target]['x'] -= fx
                    displacement[target]['y'] -= fy
            
            # Apply displacements with decreasing damping
            damping = self.damping * (1 - iteration / self.iterations)
            for node in free_nodes:
                positions[node]['x'] += displacement[node]['x'] * damping
                positions[node]['y'] += displacement[node]['y'] * damping
        
        # Return only free node positions
        return {node: positions[node] for node in free_nodes}
    
    def compute_bounding_box(
        self, 
        positions: Dict[str, Dict[str, float]]
    ) -> Tuple[float, float, float, float]:
        """
        Compute bounding box of positions.
        
        Args:
            positions: Position dictionary {node: {x, y}}
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not positions:
            return (0.0, 0.0, 1000.0, 1000.0)
        
        xs = [p['x'] for p in positions.values()]
        ys = [p['y'] for p in positions.values()]
        
        return (min(xs), min(ys), max(xs), max(ys))
    
    def merge_into_master(
        self,
        master_layout: Dict[str, Dict[str, Any]],
        new_positions: Dict[str, Dict[str, float]],
        snapshot_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge new positions into master layout.
        
        Args:
            master_layout: Existing master layout
            new_positions: New positions to add
            snapshot_id: ID of snapshot where nodes were positioned
            
        Returns:
            Updated master layout
        """
        updated = dict(master_layout)
        
        for node_id, pos in new_positions.items():
            if node_id not in updated:
                updated[node_id] = {
                    'x': pos['x'],
                    'y': pos['y'],
                    'first_seen': snapshot_id
                }
        
        return updated