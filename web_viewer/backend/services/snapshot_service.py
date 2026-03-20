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
from .duckdb_service import DuckDBService

_db = DuckDBService()

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
from .cache_service import CacheService

# Conditional imports
try:
    from engines.metrics import MetricEngine, METRIC_CATEGORIES, METRIC_PRESETS
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
        layout_service: Optional[SnapshotLayout] = None,
        unified_layout: Optional['UnifiedLayoutService'] = None
    ):
        """
        Initialize snapshot service.

        Args:
            storage: SnapshotStorage instance (created if None)
            layout_service: SnapshotLayout instance (created if None)
            unified_layout: UnifiedLayoutService instance (optional)
        """
        self.storage = storage or SnapshotStorage()
        self.layout_service = layout_service or SnapshotLayout()
        self._db_engine = None

        # Unified layout service for integrated position resolution
        # If provided, will use it to resolve positions from live/cached layouts
        self.unified_layout = unified_layout
        self._cache_service = CacheService() if unified_layout is None else None
    
    def _get_db_engine(self):
        """Legacy stub — DuckDB handles database connections now."""
        return None
    
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
        1. Check if snapshot exists â†’ return existing
        2. Execute parameterized SQL â†’ edges DataFrame
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
        source_nodes = edges_df['source'].astype(str)
        target_nodes = edges_df['target'].astype(str)
        snapshot_nodes = set(source_nodes) | set(target_nodes)
        edges_list = list(zip(source_nodes.tolist(), target_nodes.tolist()))
        
        print(f"[SNAPSHOT] Loaded {len(edges_df)} edges, {len(snapshot_nodes)} nodes")
        
        report_progress("sql", 30, f"Loaded {len(edges_df)} edges")
        
        # Get block timestamp
        block_timestamp = self._get_block_timestamp(request.block_number)
        
        # Load or initialize master layout
        report_progress("layout", 40, "Loading master layout...")

        # Derive layout for snapshot
        report_progress("layout", 50, "Computing node positions...")

        # Use unified layout service if available (integrates live positions from cosmos.gl)
        if self.unified_layout and settings.UNIFIED_LAYOUT_ENABLED:
            snapshot_layout, unknown_nodes = self.unified_layout.resolve_positions_for_snapshot(
                base_sql_file=request.base_sql_file,
                snapshot_nodes=snapshot_nodes,
                edges=edges_list,
                graph_id=request.base_sql_file
            )
        else:
            # Fallback to direct master layout derivation
            master_layout = self._get_or_initialize_master_layout(request.base_sql_file)

            snapshot_layout, unknown_nodes = self.layout_service.derive_layout(
                snapshot_nodes=snapshot_nodes,
                edges=edges_list,
                master_layout=master_layout
            )

            # Update master layout with new positions (only in non-unified mode)
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

        layout_source = LayoutSource.COMPUTED if unknown_nodes else LayoutSource.MASTER

        if unknown_nodes:
            report_progress("layout", 60, f"Positioned {len(unknown_nodes)} new nodes")
        
        # Compute metrics
        metrics_df = None
        metrics_computed = []
        
        if request.metrics_mode != MetricsMode.NONE and HAS_METRICS:
            report_progress("metrics", 70, "Computing metrics...")
            
            metrics_df, metrics_computed = self._compute_snapshot_metrics(
                edges_df, request.metrics_mode, snapshot_nodes=snapshot_nodes
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

        # Query for block numbers via DuckDB postgres_scanner
        sql = f"""
        SELECT
            date_trunc('{trunc_interval}', to_timestamp("timestamp")) as period,
            MIN("blockNumber") as block_number,
            MIN("timestamp") as timestamp
        FROM "System_Block"
        WHERE "timestamp" >= {start_timestamp}
          AND "timestamp" < {end_timestamp}
        GROUP BY period
        ORDER BY period DESC
        LIMIT {count}
        """

        try:
            df = _db.execute_postgres_sql(sql)
            rows = df.to_dict(orient='records')
        except Exception as e:
            print(f"[SNAPSHOT] Error querying block suggestions: {e}")
            return SnapshotSuggestResponse(suggestions=[])

        suggestions = []
        for row in rows:
            timestamp = row['timestamp']
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
                block_number=int(row['block_number']),
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

        try:
            df = _db.execute_postgres_sql(sql)

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
        # Note: "timestamp" column is bigint (Unix seconds)
        sql = f"""
        SELECT "timestamp"
        FROM "System_Block"
        WHERE "blockNumber" = {int(block_number)}
        LIMIT 1
        """

        try:
            df = _db.execute_postgres_sql(sql)

            if len(df) > 0:
                timestamp = df.iloc[0]['timestamp']
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
        
        # Try to load from resume layout cache
        live_layout = self._cache_service.get_resume_layout(base_sql_file) if self._cache_service else None

        if live_layout:
            print(f"[SNAPSHOT] Initializing master from resume layout cache for {base_sql_file}")
            return self.storage.initialize_master_from_live(
                base_sql_file, live_layout, "initial"
            )
        
        # No master, no live layout - return empty
        print(f"[SNAPSHOT] Warning: No layout available for {base_sql_file}")
        return {}
    
    def _compute_snapshot_metrics(
        self,
        edges_df: pd.DataFrame,
        metrics_mode: MetricsMode,
        snapshot_nodes: Optional[Set[str]] = None
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
        
        df = edges_df[['source', 'target']].copy()
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)
        G = nx.from_pandas_edgelist(
            df,
            source='source',
            target='target',
            create_using=nx.DiGraph,
        )
        if snapshot_nodes is not None:
            G.add_nodes_from(str(node_id) for node_id in snapshot_nodes)
        
        print(f"[SNAPSHOT] Computing metrics for {G.number_of_nodes()} nodes")
        
        try:
            # Map MetricsMode to mode string
            if metrics_mode == MetricsMode.BASIC:
                mode = 'basic'
            elif metrics_mode == MetricsMode.STANDARD:
                mode = settings.DEFAULT_METRICS_MODE
            elif metrics_mode == MetricsMode.FULL:
                mode = 'all'
            else:
                return None, []
            
            print(f"[SNAPSHOT] Using metrics mode: {mode}")
            
            # Parse mode to preset or categories
            preset = None
            categories = None
            
            if mode in METRIC_PRESETS:
                preset = mode
            elif ',' in mode:
                categories = [c.strip() for c in mode.split(',') if c.strip() in METRIC_CATEGORIES]
                if not categories:
                    preset = "basic"
            elif mode in METRIC_CATEGORIES:
                categories = [mode]
            else:
                preset = "basic"
            
            # Compute metrics using MetricEngine
            engine = MetricEngine(G)
            metrics_df = engine.compute(preset=preset, categories=categories)
            
            if metrics_df is not None and len(metrics_df) > 0:
                # Ensure avatar column exists
                if 'avatar' not in metrics_df.columns:
                    metrics_df['avatar'] = metrics_df.index.astype(str)
                
                metric_names = [col for col in metrics_df.columns if col != 'avatar']
                return metrics_df, metric_names
                
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Metrics computation failed: {e}")
        
        return None, []
        
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

        from_dir = self.storage.get_snapshot_dir(base_sql_file, from_block)
        to_dir = self.storage.get_snapshot_dir(base_sql_file, to_block)
        from_edges_path = from_dir / "edges.parquet"
        to_edges_path = to_dir / "edges.parquet"
        from_layout_path = from_dir / "layout.parquet"
        to_layout_path = to_dir / "layout.parquet"

        with _db.session() as session:
            if from_edges_path.exists():
                session.execute(f"""
                    CREATE TEMP TABLE from_edges AS
                    SELECT CAST(source AS VARCHAR) AS source,
                           CAST(target AS VARCHAR) AS target
                    FROM read_parquet('{from_edges_path}')
                """)
            else:
                self._materialize_snapshot_edges_temp(
                    session, "from_edges", base_sql_file, from_block
                )

            if to_edges_path.exists():
                session.execute(f"""
                    CREATE TEMP TABLE to_edges AS
                    SELECT CAST(source AS VARCHAR) AS source,
                           CAST(target AS VARCHAR) AS target
                    FROM read_parquet('{to_edges_path}')
                """)
            else:
                self._materialize_snapshot_edges_temp(
                    session, "to_edges", base_sql_file, to_block
                )

            added_edges_df = session.execute("""
                SELECT source, target FROM to_edges
                EXCEPT
                SELECT source, target FROM from_edges
            """).fetchdf()
            removed_edges_df = session.execute("""
                SELECT source, target FROM from_edges
                EXCEPT
                SELECT source, target FROM to_edges
            """).fetchdf()

            node_diff_df = session.execute(f"""
                WITH from_nodes AS (
                    SELECT CAST(node_id AS VARCHAR) AS node_id,
                           CAST(x AS DOUBLE) AS x,
                           CAST(y AS DOUBLE) AS y
                    FROM read_parquet('{from_layout_path}')
                ),
                to_nodes AS (
                    SELECT CAST(node_id AS VARCHAR) AS node_id,
                           CAST(x AS DOUBLE) AS x,
                           CAST(y AS DOUBLE) AS y
                    FROM read_parquet('{to_layout_path}')
                )
                SELECT
                    COALESCE(f.node_id, t.node_id) AS node_id,
                    f.x AS from_x,
                    f.y AS from_y,
                    t.x AS to_x,
                    t.y AS to_y,
                    CASE
                        WHEN f.node_id IS NULL THEN 'added'
                        WHEN t.node_id IS NULL THEN 'removed'
                        ELSE 'retained'
                    END AS diff_type
                FROM from_nodes f
                FULL OUTER JOIN to_nodes t
                    ON f.node_id = t.node_id
            """).fetchdf()

        added_nodes = set(
            node_diff_df.loc[node_diff_df['diff_type'] == 'added', 'node_id'].astype(str).tolist()
        )
        removed_nodes = set(
            node_diff_df.loc[node_diff_df['diff_type'] == 'removed', 'node_id'].astype(str).tolist()
        )
        retained_nodes = set(
            node_diff_df.loc[node_diff_df['diff_type'] == 'retained', 'node_id'].astype(str).tolist()
        )

        added_edges = set(zip(
            added_edges_df['source'].astype(str).tolist(),
            added_edges_df['target'].astype(str).tolist(),
        ))
        removed_edges = set(zip(
            removed_edges_df['source'].astype(str).tolist(),
            removed_edges_df['target'].astype(str).tolist(),
        ))

        merged_layout = {}
        for record in node_diff_df.to_dict(orient='records'):
            node_id = str(record['node_id'])
            if record['diff_type'] == 'removed':
                merged_layout[node_id] = {
                    'x': float(record['from_x']),
                    'y': float(record['from_y'])
                }
            else:
                merged_layout[node_id] = {
                    'x': float(record['to_x']),
                    'y': float(record['to_y'])
                }

        # Get metadata
        from_meta = self.storage.load_snapshot_metadata(base_sql_file, from_block)
        to_meta = self.storage.load_snapshot_metadata(base_sql_file, to_block)
        
        return {
            "from_snapshot": {
                "snapshot_id": f"{base_sql_file}_block_{from_block}",
                "block_number": from_block,
                "block_timestamp": from_meta.block_timestamp.isoformat() if from_meta and from_meta.block_timestamp else None,
                "node_count": from_meta.node_count if from_meta else len(retained_nodes | removed_nodes),
                "edge_count": from_meta.edge_count if from_meta else 0
            },
            "to_snapshot": {
                "snapshot_id": f"{base_sql_file}_block_{to_block}",
                "block_number": to_block,
                "block_timestamp": to_meta.block_timestamp.isoformat() if to_meta and to_meta.block_timestamp else None,
                "node_count": to_meta.node_count if to_meta else len(retained_nodes | added_nodes),
                "edge_count": to_meta.edge_count if to_meta else 0
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

    def _materialize_snapshot_edges_temp(
        self,
        session,
        table_name: str,
        base_sql_file: str,
        block_number: int
    ) -> None:
        """Load reconstructed snapshot edges into a DuckDB temp table."""
        snapshot = self.storage.load_snapshot_with_diff(base_sql_file, block_number)
        if snapshot is None:
            raise ValueError(
                f"Could not reconstruct snapshot {base_sql_file}_block_{block_number}"
            )

        rows = []
        for edge in snapshot.edges:
            if isinstance(edge, dict):
                rows.append({
                    'source': str(edge.get('source', '')),
                    'target': str(edge.get('target', ''))
                })
            else:
                source, target = edge
                rows.append({'source': str(source), 'target': str(target)})

        edges_df = pd.DataFrame(rows, columns=['source', 'target'])
        session.conn.register(f"{table_name}_df", edges_df)
        session.execute(f"""
            CREATE TEMP TABLE {table_name} AS
            SELECT CAST(source AS VARCHAR) AS source,
                   CAST(target AS VARCHAR) AS target
            FROM {table_name}_df
        """)
    
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
