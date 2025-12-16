"""
Snapshot Service

Main orchestration service for historical snapshot operations including:
- Creating snapshots from parameterized SQL
- Managing master layouts
- Computing metrics for snapshots
- Suggesting block numbers
"""

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Set

import pandas as pd
import numpy as np
import networkx as nx
from sqlalchemy import create_engine, text

from ..config import settings, HAS_ANOMALY
from ..models.snapshot import (
    SnapshotInfo,
    SnapshotData,
    SnapshotMetadata,
    SnapshotStatus,
    MetricsMode,
    LayoutSource,
    SnapshotCreateRequest,
    SnapshotBatchRequest,
    SnapshotSuggestRequest,
    BlockSuggestion,
    SnapshotSuggestResponse,
    SnapshotListResponse,
    SnapshotProgress,
)
from .snapshot_storage import SnapshotStorage
from .snapshot_layout import SnapshotLayout

# Conditional imports
try:
    from engines.graph_metrics import GraphMetrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False


class SnapshotService:
    """
    Main service for snapshot operations.
    
    Orchestrates:
    - SQL execution with block filters
    - Layout derivation from master
    - Metrics computation
    - Storage operations
    """
    
    def __init__(
        self,
        storage: Optional[SnapshotStorage] = None,
        layout_service: Optional[SnapshotLayout] = None
    ):
        """
        Initialize snapshot service.
        
        Args:
            storage: SnapshotStorage instance (created if None)
            layout_service: SnapshotLayout instance (created if None)
        """
        self.storage = storage or SnapshotStorage()
        self.layout_service = layout_service or SnapshotLayout()
        self._db_engine = None
    
    def _get_db_engine(self):
        """Get or create database engine."""
        if self._db_engine is None:
            self._db_engine = create_engine(settings.database_url)
        return self._db_engine
    
    # =========================================================================
    # Snapshot CRUD
    # =========================================================================
    
    def create_snapshot(
        self,
        request: SnapshotCreateRequest,
        progress_callback: Optional[Callable[[SnapshotProgress], None]] = None
    ) -> SnapshotInfo:
        """
        Create a single snapshot.
        
        Process:
        1. Check if snapshot exists → return existing
        2. Execute parameterized SQL → edges DataFrame
        3. Get block timestamp
        4. Load or initialize master layout
        5. Derive layout for snapshot
        6. Update master layout with new positions
        7. Compute metrics (if requested)
        8. Save snapshot
        9. Update index
        
        Args:
            request: Snapshot creation request
            progress_callback: Optional callback for progress updates
            
        Returns:
            SnapshotInfo of created/existing snapshot
        """
        start_time = time.time()
        snapshot_id = self.storage.get_snapshot_id(request.base_sql_file, request.block_number)
        
        def report_progress(stage: str, percent: int, message: str):
            if progress_callback:
                progress_callback(SnapshotProgress(
                    snapshot_id=snapshot_id,
                    status=SnapshotStatus.COMPUTING,
                    stage=stage,
                    progress_percent=percent,
                    message=message
                ))
        
        # Check if snapshot already exists
        if self.storage.snapshot_exists(request.base_sql_file, request.block_number):
            print(f"[SNAPSHOT] Snapshot already exists: {snapshot_id}")
            existing = self.storage.load_snapshot_metadata(
                request.base_sql_file, request.block_number
            )
            if existing:
                return existing
        
        report_progress("sql", 10, "Executing SQL query...")
        
        # Execute parameterized SQL
        edges_df = self._execute_snapshot_sql(request.base_sql_file, request.block_number)
        
        if edges_df.empty:
            raise ValueError(f"No edges found for block {request.block_number}")
        
        # Extract unique nodes
        snapshot_nodes = self._extract_nodes(edges_df)
        edges_list = [(str(row['source']), str(row['target'])) for _, row in edges_df.iterrows()]
        
        print(f"[SNAPSHOT] Loaded {len(edges_df)} edges, {len(snapshot_nodes)} nodes")
        
        report_progress("sql", 30, f"Loaded {len(edges_df)} edges")
        
        # Get block timestamp
        block_timestamp = self._get_block_timestamp(request.block_number)
        
        # Load or initialize master layout
        report_progress("layout", 40, "Loading master layout...")
        
        master_layout = self._get_or_initialize_master_layout(request.base_sql_file)
        
        # Derive layout for snapshot
        report_progress("layout", 50, "Computing node positions...")
        
        snapshot_layout, unknown_nodes = self.layout_service.derive_layout(
            snapshot_nodes=snapshot_nodes,
            edges=edges_list,
            master_layout=master_layout
        )
        
        layout_source = LayoutSource.COMPUTED if unknown_nodes else LayoutSource.MASTER
        
        # Update master layout with new positions
        if unknown_nodes:
            report_progress("layout", 60, f"Updating master layout (+{len(unknown_nodes)} nodes)...")
            
            new_positions = {
                node: snapshot_layout[node]
                for node in unknown_nodes
            }
            master_layout = self.layout_service.merge_into_master(
                master_layout, new_positions, snapshot_id
            )
            self.storage.save_master_layout(request.base_sql_file, master_layout)
        
        # Compute metrics
        metrics_df = None
        metrics_computed = []
        
        if request.metrics_mode != MetricsMode.NONE and HAS_METRICS:
            report_progress("metrics", 70, "Computing metrics...")
            
            metrics_df, metrics_computed = self._compute_snapshot_metrics(
                edges_df, request.metrics_mode
            )
        
        report_progress("saving", 90, "Saving snapshot...")
        
        # Create metadata
        computation_time = time.time() - start_time
        created_at = datetime.utcnow()
        
        # Generate label if not provided
        label = request.label
        if not label and block_timestamp:
            label = block_timestamp.strftime("%Y-%m-%d (%A)")
        
        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            base_sql_file=request.base_sql_file,
            block_number=request.block_number,
            block_timestamp=block_timestamp,
            label=label,
            node_count=len(snapshot_nodes),
            edge_count=len(edges_df),
            metrics_computed=metrics_computed,
            metrics_mode=request.metrics_mode.value,
            layout_source=layout_source.value,
            layout_unknown_nodes=len(unknown_nodes),
            created_at=created_at,
            computation_time_seconds=computation_time
        )
        
        # Save snapshot
        self.storage.save_snapshot(
            base_sql_file=request.base_sql_file,
            block_number=request.block_number,
            edges_df=edges_df,
            layout=snapshot_layout,
            metrics_df=metrics_df,
            metadata=metadata
        )
        
        # Create SnapshotInfo for response
        snapshot_info = SnapshotInfo(
            snapshot_id=snapshot_id,
            base_sql_file=request.base_sql_file,
            block_number=request.block_number,
            block_timestamp=block_timestamp,
            label=label,
            node_count=len(snapshot_nodes),
            edge_count=len(edges_df),
            metrics_computed=metrics_computed,
            layout_source=layout_source,
            layout_unknown_nodes=len(unknown_nodes),
            created_at=created_at,
            status=SnapshotStatus.READY
        )
        
        # Update index
        self.storage.update_index(snapshot_info)
        
        report_progress("saving", 100, "Snapshot created!")
        
        print(f"[SNAPSHOT] Created {snapshot_id} in {computation_time:.2f}s")
        
        return snapshot_info
    
    def get_snapshot(self, base_sql_file: str, block_number: int) -> SnapshotData:
        """
        Get full snapshot data.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            SnapshotData with edges, layout, metrics, metadata
        """
        return self.storage.load_snapshot(base_sql_file, block_number)
    
    def get_snapshot_info(self, base_sql_file: str, block_number: int) -> Optional[SnapshotInfo]:
        """
        Get snapshot metadata only.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            SnapshotInfo or None if not found
        """
        return self.storage.load_snapshot_metadata(base_sql_file, block_number)
    
    def list_snapshots(self, base_sql_file: Optional[str] = None) -> List[SnapshotInfo]:
        """
        List available snapshots.
        
        Args:
            base_sql_file: Optional filter by SQL file
            
        Returns:
            List of SnapshotInfo objects
        """
        return self.storage.list_snapshots(base_sql_file)
    
    def delete_snapshot(self, base_sql_file: str, block_number: int) -> bool:
        """
        Delete a snapshot.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            True if deleted, False if not found
        """
        return self.storage.delete_snapshot(base_sql_file, block_number)
    
    # =========================================================================
    # Batch Operations
    # =========================================================================
    
    def create_batch(
        self,
        request: SnapshotBatchRequest,
        progress_callback: Optional[Callable[[int, int, SnapshotInfo], None]] = None
    ) -> List[SnapshotInfo]:
        """
        Create multiple snapshots.
        
        Block numbers are sorted ascending (oldest first) to build
        master layout chronologically.
        
        Args:
            request: Batch creation request
            progress_callback: Callback(current, total, snapshot_info)
            
        Returns:
            List of created SnapshotInfo objects
        """
        # Sort block numbers ascending (oldest first)
        block_numbers = sorted(request.block_numbers)
        total = len(block_numbers)
        results = []
        
        print(f"[SNAPSHOT] Starting batch creation: {total} snapshots")
        
        for idx, block_number in enumerate(block_numbers, 1):
            try:
                create_request = SnapshotCreateRequest(
                    base_sql_file=request.base_sql_file,
                    block_number=block_number,
                    metrics_mode=request.metrics_mode
                )
                
                snapshot_info = self.create_snapshot(create_request)
                results.append(snapshot_info)
                
                if progress_callback:
                    progress_callback(idx, total, snapshot_info)
                    
            except Exception as e:
                print(f"[SNAPSHOT] Error creating snapshot for block {block_number}: {e}")
                # Continue with remaining snapshots
        
        print(f"[SNAPSHOT] Batch complete: {len(results)}/{total} snapshots created")
        
        return results
    
    def suggest_block_numbers(self, request: SnapshotSuggestRequest) -> SnapshotSuggestResponse:
        """
        Suggest block numbers for snapshot creation.
        
        Queries System_Block table to find first block of each time period.
        
        Args:
            request: Suggestion request with interval, dates, count
            
        Returns:
            List of BlockSuggestion objects
        """
        engine = self._get_db_engine()
        
        # Build date range
        end_date = request.end_date or datetime.utcnow().date()
        start_date = request.start_date or (end_date - timedelta(days=30))
        count = min(request.count or 30, settings.SNAPSHOT_MAX_SUGGESTIONS)
        
        # Map interval to PostgreSQL date_trunc argument
        interval_map = {
            'daily': 'day',
            'weekly': 'week',
            'monthly': 'month'
        }
        trunc_interval = interval_map.get(request.interval, 'day')
        
        # Convert dates to Unix timestamps for comparison
        start_dt = datetime(start_date.year, start_date.month, start_date.day)
        end_dt = datetime(end_date.year, end_date.month, end_date.day) + timedelta(days=1)
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        
        # Query for block numbers
        # Note: "timestamp" column is bigint (Unix seconds)
        sql = f"""
        SELECT 
            date_trunc('{trunc_interval}', to_timestamp("timestamp")) as period,
            MIN("blockNumber") as block_number,
            MIN("timestamp") as timestamp
        FROM "System_Block"
        WHERE "timestamp" >= :start_timestamp
          AND "timestamp" < :end_timestamp
        GROUP BY period
        ORDER BY period DESC
        LIMIT :count
        """
        
        try:
            with engine.connect() as conn:
                result = conn.execute(
                    text(sql),
                    {
                        'start_timestamp': start_timestamp,
                        'end_timestamp': end_timestamp,
                        'count': count
                    }
                )
                rows = result.fetchall()
        except Exception as e:
            print(f"[SNAPSHOT] Error querying block suggestions: {e}")
            return SnapshotSuggestResponse(suggestions=[])
        
        suggestions = []
        for row in rows:
            timestamp = row.timestamp
            # Handle bigint Unix timestamp
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.utcfromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    timestamp = datetime.utcfromtimestamp(float(timestamp))
            
            # Create human-readable label
            day_name = timestamp.strftime("%A")
            date_str = timestamp.strftime("%Y-%m-%d")
            label = f"{date_str} ({day_name})"
            
            suggestions.append(BlockSuggestion(
                block_number=int(row.block_number),
                timestamp=timestamp,
                label=label
            ))
        
        return SnapshotSuggestResponse(suggestions=suggestions)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _execute_snapshot_sql(self, base_sql_file: str, block_number: int) -> pd.DataFrame:
        """
        Execute parameterized SQL with block filter.
        
        Looks for SQL template in sql/snapshots/{base}_snapshot.sql
        and replaces {block_filter} placeholder.
        
        Args:
            base_sql_file: Base SQL file name (e.g., 'crc_v2_trusts')
            block_number: Block number to filter by
            
        Returns:
            DataFrame with source, target columns
        """
        sql_path = settings.SNAPSHOT_SQL_DIR / f"{base_sql_file}_snapshot.sql"
        
        if not sql_path.exists():
            raise ValueError(f"Snapshot SQL template not found: {sql_path}")
        
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_template = f.read()
        
        # Replace block_number placeholder directly
        sql = sql_template.replace("{block_number}", str(block_number))
        
        print(f"[SNAPSHOT] Executing SQL with block filter: blockNumber <= {block_number}")
        
        engine = self._get_db_engine()
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(sql), conn)
            
            # Ensure source/target columns exist
            if 'source' not in df.columns or 'target' not in df.columns:
                raise ValueError("SQL must return 'source' and 'target' columns")
            
            # Convert to strings and lowercase
            df['source'] = df['source'].astype(str).str.lower()
            df['target'] = df['target'].astype(str).str.lower()
            
            return df
            
        except Exception as e:
            print(f"[SNAPSHOT] SQL execution error: {e}")
            raise
    
    def _extract_nodes(self, edges_df: pd.DataFrame) -> Set[str]:
        """Extract unique node IDs from edges DataFrame."""
        sources = set(edges_df['source'].astype(str))
        targets = set(edges_df['target'].astype(str))
        return sources | targets
    
    def _get_block_timestamp(self, block_number: int) -> Optional[datetime]:
        """
        Get timestamp for a block number from System_Block table.
        
        Args:
            block_number: Block number
            
        Returns:
            Timestamp or None if not found
        """
        engine = self._get_db_engine()
        
        # Note: "timestamp" column is bigint (Unix seconds)
        sql = """
        SELECT "timestamp"
        FROM "System_Block"
        WHERE "blockNumber" = :block_number
        LIMIT 1
        """
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql), {'block_number': block_number})
                row = result.fetchone()
                
            if row:
                timestamp = row.timestamp
                # Handle bigint Unix timestamp
                if isinstance(timestamp, (int, float)):
                    return datetime.utcfromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    # Try parsing as ISO format first
                    try:
                        return datetime.fromisoformat(timestamp)
                    except ValueError:
                        # Try parsing as numeric string
                        return datetime.utcfromtimestamp(float(timestamp))
                return timestamp
                
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Could not get block timestamp: {e}")
        
        return None
    
    def _get_or_initialize_master_layout(self, base_sql_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Get existing master layout or initialize from live layout.
        
        Args:
            base_sql_file: Base SQL file name
            
        Returns:
            Master layout dictionary
        """
        # Check for existing master
        if self.storage.master_layout_exists(base_sql_file):
            return self.storage.load_master_layout(base_sql_file)
        
        # Try to load from live layout cache
        live_layout_path = settings.LAYOUTS_DIR / f"{base_sql_file}.parquet"
        
        if live_layout_path.exists():
            print(f"[SNAPSHOT] Initializing master from live layout: {live_layout_path}")
            
            try:
                df = pd.read_parquet(live_layout_path)
                live_layout = {}
                
                # Determine column names
                id_col = 'id' if 'id' in df.columns else df.columns[0]
                x_col = 'x' if 'x' in df.columns else 'position_x'
                y_col = 'y' if 'y' in df.columns else 'position_y'
                
                for _, row in df.iterrows():
                    node_id = str(row[id_col])
                    live_layout[node_id] = {
                        'x': float(row[x_col]),
                        'y': float(row[y_col])
                    }
                
                return self.storage.initialize_master_from_live(
                    base_sql_file, live_layout, "initial"
                )
                
            except Exception as e:
                print(f"[SNAPSHOT] Warning: Could not load live layout: {e}")
        
        # No master, no live layout - return empty
        print(f"[SNAPSHOT] Warning: No layout available for {base_sql_file}")
        return {}
    
    def _compute_snapshot_metrics(
        self,
        edges_df: pd.DataFrame,
        metrics_mode: MetricsMode
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        Compute metrics for snapshot graph.
        
        Args:
            edges_df: DataFrame with source, target columns
            metrics_mode: Which metrics to compute
            
        Returns:
            Tuple of (metrics DataFrame, list of metric names)
        """
        if not HAS_METRICS:
            return None, []
        
        # Build NetworkX graph
        G = nx.DiGraph()
        for _, row in edges_df.iterrows():
            G.add_edge(str(row['source']), str(row['target']))
        
        print(f"[SNAPSHOT] Computing metrics for {G.number_of_nodes()} nodes")
        
        try:
            # Map MetricsMode to GraphMetrics mode string
            # Use the app's DEFAULT_METRICS_MODE from config for STANDARD
            if metrics_mode == MetricsMode.BASIC:
                mode = 'basic'
            elif metrics_mode == MetricsMode.STANDARD:
                # Use the same default as the main app
                mode = settings.DEFAULT_METRICS_MODE
            elif metrics_mode == MetricsMode.FULL:
                mode = 'all'
            else:
                return None, []
            
            print(f"[SNAPSHOT] Using metrics mode: {mode}")
            
            # Compute metrics - mode is passed to constructor
            calculator = GraphMetrics(G, metrics_mode=mode)
            metrics_df = calculator.compute_all()
            
            if metrics_df is not None and len(metrics_df) > 0:
                # Ensure avatar column exists
                if 'avatar' not in metrics_df.columns:
                    metrics_df['avatar'] = metrics_df.index.astype(str)
                
                metric_names = [col for col in metrics_df.columns if col != 'avatar']
                return metrics_df, metric_names
                
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Metrics computation failed: {e}")
        
        return None, []
    
    # =========================================================================
    # Live Layout Access (for master initialization)
    # =========================================================================
    
    def get_live_layout(self, base_sql_file: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get current live layout positions from network service.
        
        Used when initializing master layout from current state.
        
        Args:
            base_sql_file: Base SQL file name
            
        Returns:
            Layout dictionary or None if not available
        """
        try:
            from .network_service import network_service
            
            if base_sql_file in network_service.layouts:
                return network_service.layouts[base_sql_file]
                
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Could not get live layout: {e}")
        
        return None
    
    def get_available_snapshot_sql_files(self) -> List[str]:
        """Get list of SQL files that have snapshot templates."""
        return self.storage.get_available_snapshot_sql_files()
    
    # =========================================================================
    # Comparison Methods
    # =========================================================================
    
    def compare_snapshots(
        self, 
        base_sql_file: str, 
        from_block: int, 
        to_block: int
    ) -> Dict[str, Any]:
        """
        Compare two snapshots and compute differences.
        
        Args:
            base_sql_file: Base SQL file name
            from_block: Earlier block number
            to_block: Later block number
            
        Returns:
            Dict with comparison results
        """
        print(f"[SNAPSHOT] Comparing {base_sql_file}: block {from_block} -> {to_block}")
        
        # Load node sets from both snapshots
        from_nodes = self.storage.load_snapshot_node_ids(base_sql_file, from_block)
        to_nodes = self.storage.load_snapshot_node_ids(base_sql_file, to_block)
        
        # Load edge sets
        from_edges = self.storage.load_snapshot_edge_set(base_sql_file, from_block)
        to_edges = self.storage.load_snapshot_edge_set(base_sql_file, to_block)
        
        # Compute differences
        added_nodes = to_nodes - from_nodes
        removed_nodes = from_nodes - to_nodes
        retained_nodes = from_nodes & to_nodes
        
        added_edges = to_edges - from_edges
        removed_edges = from_edges - to_edges
        
        # Load layouts for positioning comparison view
        from_layout = self.storage.load_snapshot_layout_dict(base_sql_file, from_block)
        to_layout = self.storage.load_snapshot_layout_dict(base_sql_file, to_block)
        
        # Merge layouts: use 'to' positions for added/retained, 'from' for removed
        merged_layout = {}
        for node_id in retained_nodes | added_nodes:
            if node_id in to_layout:
                merged_layout[node_id] = to_layout[node_id]
        for node_id in removed_nodes:
            if node_id in from_layout:
                merged_layout[node_id] = from_layout[node_id]
        
        # Get metadata
        from_meta = self.storage.load_snapshot_metadata(base_sql_file, from_block)
        to_meta = self.storage.load_snapshot_metadata(base_sql_file, to_block)
        
        return {
            "from_snapshot": {
                "snapshot_id": f"{base_sql_file}_block_{from_block}",
                "block_number": from_block,
                "block_timestamp": from_meta.block_timestamp.isoformat() if from_meta and from_meta.block_timestamp else None,
                "node_count": len(from_nodes),
                "edge_count": len(from_edges)
            },
            "to_snapshot": {
                "snapshot_id": f"{base_sql_file}_block_{to_block}",
                "block_number": to_block,
                "block_timestamp": to_meta.block_timestamp.isoformat() if to_meta and to_meta.block_timestamp else None,
                "node_count": len(to_nodes),
                "edge_count": len(to_edges)
            },
            "diff": {
                "added_nodes": list(added_nodes),
                "removed_nodes": list(removed_nodes),
                "retained_nodes": list(retained_nodes),
                "added_node_count": len(added_nodes),
                "removed_node_count": len(removed_nodes),
                "retained_node_count": len(retained_nodes),
                "added_edge_count": len(added_edges),
                "removed_edge_count": len(removed_edges)
            },
            "layout": merged_layout,
            # Edge lists for rendering (only return counts, not full lists for large graphs)
            "edges": {
                "added": [{"source": e[0], "target": e[1]} for e in list(added_edges)[:1000]],
                "removed": [{"source": e[0], "target": e[1]} for e in list(removed_edges)[:1000]],
                "added_truncated": len(added_edges) > 1000,
                "removed_truncated": len(removed_edges) > 1000
            }
        }
    
    # =========================================================================
    # Animation Methods
    # =========================================================================
    
    def get_animation_data(self, base_sql_file: str) -> Dict[str, Any]:
        """
        Get all snapshots with compact layout data for animation.
        
        Args:
            base_sql_file: Base SQL file name
            
        Returns:
            Dict with animation-ready data
        """
        print(f"[SNAPSHOT] Loading animation data for {base_sql_file}")
        
        # Get all snapshots
        snapshots = self.storage.list_snapshots(base_sql_file)
        
        if not snapshots:
            return {
                "base_sql_file": base_sql_file,
                "snapshots": [],
                "total": 0
            }
        
        # Sort by block number
        snapshots.sort(key=lambda s: s.block_number)
        
        # Load layouts for each snapshot
        frames = []
        all_nodes = set()
        
        for snapshot in snapshots:
            layout = self.storage.load_snapshot_layout_dict(
                base_sql_file, snapshot.block_number
            )
            node_ids = self.storage.load_snapshot_node_ids(
                base_sql_file, snapshot.block_number
            )
            
            all_nodes.update(node_ids)
            
            frames.append({
                "snapshot_id": snapshot.snapshot_id,
                "block_number": snapshot.block_number,
                "block_timestamp": snapshot.block_timestamp.isoformat() if snapshot.block_timestamp else None,
                "label": snapshot.label,
                "node_count": snapshot.node_count,
                "edge_count": snapshot.edge_count,
                "layout": layout,
                "node_ids": list(node_ids)
            })
        
        return {
            "base_sql_file": base_sql_file,
            "frames": frames,
            "total": len(frames),
            "all_nodes": list(all_nodes)
        }


# Singleton instance
snapshot_service = SnapshotService()