"""
Unified Layout Service

Single source of truth for all node positions across live graphs and snapshots.
This service bridges the gap between:
- Live graph layouts (from network_service)
- Cached layouts (from cache_service)
- Master layouts (from snapshot_storage)
- Spring-computed positions (for new nodes)

The position resolution hierarchy (highest to lowest priority):
1. Live cosmos.gl positions (synced from frontend)
2. Cached layout positions (user-saved)
3. Master layout positions (snapshot-derived)
4. Spring-computed positions (for new nodes)
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

from ..config import settings
from .cache_service import CacheService
from .snapshot_storage import SnapshotStorage
from .snapshot_layout import SnapshotLayout


class UnifiedLayoutService:
    """
    Single source of truth for all layouts.

    Ensures consistency between:
    - Live graph viewing
    - Historical snapshots
    - Animation transitions
    - cosmos.gl browser layouts
    """

    def __init__(
        self,
        cache_service: CacheService,
        snapshot_storage: SnapshotStorage
    ):
        """
        Initialize the unified layout service.

        Args:
            cache_service: Service for layout cache operations
            snapshot_storage: Service for snapshot master layouts
        """
        self.cache = cache_service
        self.snapshots = snapshot_storage
        self.spring_layout = SnapshotLayout()

        # In-memory live positions from frontend (cosmos.gl)
        # Structure: {graph_id: {node_id: {x, y}}}
        self._live_positions: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Track sync metadata
        # Structure: {graph_id: {last_sync: datetime, source: str, count: int}}
        self._sync_metadata: Dict[str, Dict[str, Any]] = {}

    # =========================================================================
    # Position Resolution (Read Operations)
    # =========================================================================

    def _ensure_hydrated(self, graph_id: str) -> None:
        """Hydrate in-memory live positions from persisted cosmos state on demand."""
        if graph_id in self._live_positions and self._live_positions[graph_id]:
            return

        persisted = self.cache.get_cosmos_live_layout(graph_id)
        if not persisted:
            return

        self._live_positions[graph_id] = persisted
        self._sync_metadata[graph_id] = {
            'last_sync': datetime.utcnow(),
            'source': 'cosmos_hydrated',
            'count': len(persisted),
            'total_nodes': len(persisted)
        }
        print(f"[UNIFIED_LAYOUT] Hydrated {len(persisted)} persisted cosmos positions for {graph_id}")

    def get_position(
        self,
        graph_id: str,
        node_id: str
    ) -> Optional[Dict[str, float]]:
        """
        Get best available position for a single node.

        Resolution order:
        1. Live positions (from cosmos.gl)
        2. Cached layout
        3. Master layout (for snapshot base_sql_file)

        Args:
            graph_id: Graph identifier
            node_id: Node identifier

        Returns:
            Position dict {x, y} or None if not found
        """
        self._ensure_hydrated(graph_id)

        # 1. Check live positions
        if graph_id in self._live_positions:
            live = self._live_positions[graph_id]
            if node_id in live:
                return {'x': live[node_id]['x'], 'y': live[node_id]['y']}

        # 2. Check cached layout
        cached = self.cache.get_cached_layout(graph_id)
        if cached and node_id in cached:
            return {'x': cached[node_id]['x'], 'y': cached[node_id]['y']}

        # 3. Check master layout (graph_id may be the base_sql_file)
        if self.snapshots.master_layout_exists(graph_id):
            master = self.snapshots.load_master_layout(graph_id)
            if node_id in master:
                return {'x': master[node_id]['x'], 'y': master[node_id]['y']}

        return None

    def get_positions_batch(
        self,
        graph_id: str,
        node_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get positions for multiple nodes efficiently.

        Args:
            graph_id: Graph identifier
            node_ids: List of node identifiers

        Returns:
            Dict mapping node_id -> {x, y} (only for found nodes)
        """
        self._ensure_hydrated(graph_id)

        result: Dict[str, Dict[str, float]] = {}
        remaining = set(node_ids)

        # 1. Check live positions
        if graph_id in self._live_positions:
            live = self._live_positions[graph_id]
            for node_id in list(remaining):
                if node_id in live:
                    result[node_id] = {
                        'x': live[node_id]['x'],
                        'y': live[node_id]['y']
                    }
                    remaining.discard(node_id)

        if not remaining:
            return result

        # 2. Check cached layout
        cached = self.cache.get_cached_layout(graph_id)
        if cached:
            for node_id in list(remaining):
                if node_id in cached:
                    result[node_id] = {
                        'x': cached[node_id]['x'],
                        'y': cached[node_id]['y']
                    }
                    remaining.discard(node_id)

        if not remaining:
            return result

        # 3. Check master layout
        if self.snapshots.master_layout_exists(graph_id):
            master = self.snapshots.load_master_layout(graph_id)
            for node_id in list(remaining):
                if node_id in master:
                    result[node_id] = {
                        'x': master[node_id]['x'],
                        'y': master[node_id]['y']
                    }
                    remaining.discard(node_id)

        return result

    def get_all_positions(self, graph_id: str) -> Dict[str, Dict[str, float]]:
        """
        Get all available positions for a graph.

        Merges from all sources with live taking precedence.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict mapping node_id -> {x, y}
        """
        self._ensure_hydrated(graph_id)

        result: Dict[str, Dict[str, float]] = {}

        # Start with master (lowest priority)
        if self.snapshots.master_layout_exists(graph_id):
            master = self.snapshots.load_master_layout(graph_id)
            for node_id, pos in master.items():
                result[node_id] = {'x': pos['x'], 'y': pos['y']}

        # Override with cached
        cached = self.cache.get_cached_layout(graph_id)
        if cached:
            for node_id, pos in cached.items():
                result[node_id] = {'x': pos['x'], 'y': pos['y']}

        # Override with live (highest priority)
        if graph_id in self._live_positions:
            for node_id, pos in self._live_positions[graph_id].items():
                result[node_id] = {'x': pos['x'], 'y': pos['y']}

        return result

    # =========================================================================
    # Position Updates (Write Operations)
    # =========================================================================

    def update_live_positions(
        self,
        graph_id: str,
        positions: Dict[str, Dict[str, float]],
        source: str = "cosmos"
    ) -> int:
        """
        Update live positions from frontend.

        This is called when cosmos.gl syncs positions to the server.

        Args:
            graph_id: Graph identifier
            positions: Position updates {node_id: {x, y}}
            source: Source identifier (e.g., "cosmos", "manual")

        Returns:
            Number of positions updated
        """
        if not positions:
            return 0

        if graph_id not in self._live_positions:
            self._live_positions[graph_id] = {}

        # Update positions
        for node_id, pos in positions.items():
            self._live_positions[graph_id][node_id] = {
                'x': float(pos['x']),
                'y': float(pos['y'])
            }

        # Update metadata
        self._sync_metadata[graph_id] = {
            'last_sync': datetime.utcnow(),
            'source': source,
            'count': len(positions),
            'total_nodes': len(self._live_positions[graph_id])
        }

        print(f"[UNIFIED_LAYOUT] Updated {len(positions)} live positions "
              f"for {graph_id} from {source}")

        return len(positions)

    def clear_live_positions(self, graph_id: str) -> None:
        """
        Clear live positions for a graph.

        Called when graph is unloaded or reset.

        Args:
            graph_id: Graph identifier
        """
        if graph_id in self._live_positions:
            del self._live_positions[graph_id]
            print(f"[UNIFIED_LAYOUT] Cleared live positions for {graph_id}")

        if graph_id in self._sync_metadata:
            del self._sync_metadata[graph_id]

    # =========================================================================
    # Master Layout Sync
    # =========================================================================

    def sync_to_master(
        self,
        graph_id: str,
        base_sql_file: str
    ) -> int:
        """
        Sync current positions to master layout.

        This ensures that cosmos.gl positions are used when creating
        new snapshots, maintaining visual consistency.

        Args:
            graph_id: Graph identifier (source of positions)
            base_sql_file: SQL file name (destination for master)

        Returns:
            Number of positions synced
        """
        # Get all current positions for the graph
        all_positions = self.get_all_positions(graph_id)

        if not all_positions:
            print(f"[UNIFIED_LAYOUT] No positions to sync for {graph_id}")
            return 0

        # Load existing master
        master = self.snapshots.load_master_layout(base_sql_file)

        # Merge positions into master
        synced = 0
        for node_id, pos in all_positions.items():
            if node_id not in master:
                master[node_id] = {
                    'x': pos['x'],
                    'y': pos['y'],
                    'first_seen': 'live_sync'
                }
                synced += 1
            else:
                # Update existing position if from live source
                if graph_id in self._live_positions:
                    if node_id in self._live_positions[graph_id]:
                        master[node_id]['x'] = pos['x']
                        master[node_id]['y'] = pos['y']
                        synced += 1

        # Save updated master
        if synced > 0:
            self.snapshots.save_master_layout(base_sql_file, master)
            print(f"[UNIFIED_LAYOUT] Synced {synced} positions to master "
                  f"for {base_sql_file}")

        return synced

    def sync_all_to_master(
        self,
        graph_id: str,
        base_sql_file: str,
        positions: Dict[str, Dict[str, float]]
    ) -> int:
        """
        Sync all provided positions to master layout.

        This is a full sync that updates or adds all provided positions.

        Args:
            graph_id: Graph identifier (for logging)
            base_sql_file: SQL file name
            positions: Positions to sync {node_id: {x, y}}

        Returns:
            Number of positions synced
        """
        if not positions:
            return 0

        # Load existing master or create new
        master = self.snapshots.load_master_layout(base_sql_file)

        updated = 0
        added = 0

        for node_id, pos in positions.items():
            if node_id in master:
                # Update existing
                master[node_id]['x'] = pos['x']
                master[node_id]['y'] = pos['y']
                updated += 1
            else:
                # Add new
                master[node_id] = {
                    'x': pos['x'],
                    'y': pos['y'],
                    'first_seen': 'cosmos_sync'
                }
                added += 1

        # Save updated master
        self.snapshots.save_master_layout(base_sql_file, master)

        print(f"[UNIFIED_LAYOUT] Full sync to master for {base_sql_file}: "
              f"{updated} updated, {added} added")

        return updated + added

    # =========================================================================
    # Snapshot Layout Resolution
    # =========================================================================

    def resolve_positions_for_snapshot(
        self,
        base_sql_file: str,
        snapshot_nodes: Set[str],
        edges: List[Tuple[str, str]],
        graph_id: Optional[str] = None
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """
        Resolve positions for a snapshot using all available sources.

        This method is the primary integration point for snapshot creation.
        It uses the unified layout to find known positions and falls back
        to spring layout for unknown nodes.

        Args:
            base_sql_file: SQL file name for master layout
            snapshot_nodes: Set of nodes in the snapshot
            edges: List of (source, target) tuples
            graph_id: Optional graph ID for live positions

        Returns:
            Tuple of:
            - layout_dict: {node_id: {x, y}} for all snapshot nodes
            - unknown_nodes: List of nodes that needed spring computation
        """
        # Build combined positions from all sources
        known_positions: Dict[str, Dict[str, float]] = {}
        unknown_nodes: Set[str] = set()

        # 1. Check live positions if graph_id provided
        if graph_id and graph_id in self._live_positions:
            live = self._live_positions[graph_id]
            for node_id in snapshot_nodes:
                if node_id in live:
                    known_positions[node_id] = {
                        'x': live[node_id]['x'],
                        'y': live[node_id]['y']
                    }

        # 2. Check cached layout
        cached = self.cache.get_cached_layout(graph_id or base_sql_file)
        if cached:
            for node_id in snapshot_nodes:
                if node_id not in known_positions and node_id in cached:
                    known_positions[node_id] = {
                        'x': cached[node_id]['x'],
                        'y': cached[node_id]['y']
                    }

        # 3. Check master layout
        master = self.snapshots.load_master_layout(base_sql_file)
        if master:
            for node_id in snapshot_nodes:
                if node_id not in known_positions and node_id in master:
                    known_positions[node_id] = {
                        'x': master[node_id]['x'],
                        'y': master[node_id]['y']
                    }

        # Identify unknown nodes
        unknown_nodes = snapshot_nodes - set(known_positions.keys())

        print(f"[UNIFIED_LAYOUT] Resolving snapshot positions: "
              f"{len(snapshot_nodes)} nodes, {len(known_positions)} known, "
              f"{len(unknown_nodes)} unknown")

        # If no unknown nodes, return known positions directly
        if not unknown_nodes:
            return known_positions, []

        # Use the fast igraph-based positioning path when available.
        try:
            new_positions = self.spring_layout.position_unknown_nodes_fast(
                edges=edges,
                fixed_positions=known_positions,
                free_nodes=unknown_nodes,
                algorithm=getattr(settings, 'SNAPSHOT_LAYOUT_ALGORITHM', 'auto')
            )
        except Exception:
            new_positions = self.spring_layout.position_unknown_nodes(
                edges=edges,
                fixed_positions=known_positions,
                free_nodes=unknown_nodes
            )

        # Combine all positions
        layout = dict(known_positions)
        layout.update(new_positions)

        # Update master with new positions
        if new_positions:
            updated_master = self.spring_layout.merge_into_master(
                master_layout=master,
                new_positions=new_positions,
                snapshot_id=f"{base_sql_file}_spring_derived"
            )
            self.snapshots.save_master_layout(base_sql_file, updated_master)

        return layout, list(unknown_nodes)

    def resolve_new_nodes(
        self,
        graph_id: str,
        new_nodes: Set[str],
        edges: List[Tuple[str, str]],
        existing_positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Resolve positions for newly added nodes (e.g., during auto-reload).

        This method is used when new nodes are added to a live graph and
        need positions that don't disrupt the existing layout.

        Args:
            graph_id: Graph identifier
            new_nodes: Set of new node IDs
            edges: All edges (including those involving new nodes)
            existing_positions: Current positions of existing nodes

        Returns:
            Positions for new nodes only {node_id: {x, y}}
        """
        if not new_nodes:
            return {}

        # First check if any new nodes have positions in master
        known_positions: Dict[str, Dict[str, float]] = {}
        unknown_nodes: Set[str] = set(new_nodes)

        # Check master layout
        if self.snapshots.master_layout_exists(graph_id):
            master = self.snapshots.load_master_layout(graph_id)
            for node_id in new_nodes:
                if node_id in master:
                    known_positions[node_id] = {
                        'x': master[node_id]['x'],
                        'y': master[node_id]['y']
                    }
                    unknown_nodes.discard(node_id)

        # If all nodes have master positions, return them
        if not unknown_nodes:
            print(f"[UNIFIED_LAYOUT] All {len(new_nodes)} new nodes found in master")
            return known_positions

        # Use spring layout for truly unknown nodes
        # Combine existing positions with any known new positions as fixed
        fixed_positions = dict(existing_positions)
        fixed_positions.update(known_positions)

        # Filter edges to only those involving unknown nodes
        relevant_edges = [
            (s, t) for s, t in edges
            if s in unknown_nodes or t in unknown_nodes
        ]

        # Use shorter iterations for speed (continuous mode needs to be fast)
        quick_spring = SnapshotLayout(iterations=25)
        try:
            new_positions = quick_spring.position_unknown_nodes_fast(
                edges=relevant_edges,
                fixed_positions=fixed_positions,
                free_nodes=unknown_nodes,
                algorithm='auto'
            )
        except Exception:
            new_positions = quick_spring.position_unknown_nodes(
                edges=relevant_edges,
                fixed_positions=fixed_positions,
                free_nodes=unknown_nodes
            )

        # Combine known and computed
        result = dict(known_positions)
        result.update(new_positions)

        print(f"[UNIFIED_LAYOUT] Resolved {len(new_nodes)} new nodes: "
              f"{len(known_positions)} from master, {len(new_positions)} computed")

        return result

    # =========================================================================
    # Status and Metadata
    # =========================================================================

    def get_sync_status(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """
        Get sync status for a graph.

        Args:
            graph_id: Graph identifier

        Returns:
            Sync metadata or None if no sync data
        """
        return self._sync_metadata.get(graph_id)

    def get_live_position_count(self, graph_id: str) -> int:
        """
        Get count of live positions for a graph.

        Args:
            graph_id: Graph identifier

        Returns:
            Number of live positions
        """
        if graph_id in self._live_positions:
            return len(self._live_positions[graph_id])
        return 0

    def has_live_positions(self, graph_id: str) -> bool:
        """
        Check if live positions exist for a graph.

        Args:
            graph_id: Graph identifier

        Returns:
            True if live positions exist
        """
        return graph_id in self._live_positions and len(self._live_positions[graph_id]) > 0
