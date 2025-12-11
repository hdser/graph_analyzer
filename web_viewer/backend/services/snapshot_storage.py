"""
Snapshot Storage Service

Handles all file I/O operations for historical snapshots including:
- Index management
- Master layout operations
- Snapshot CRUD operations
"""

import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import settings
from ..models.snapshot import (
    SnapshotInfo,
    SnapshotData,
    SnapshotMetadata,
    SnapshotStatus,
    LayoutSource,
    IndexEntry,
    StorageStats,
)


class SnapshotStorage:
    """
    Service for snapshot file I/O operations.
    
    Manages:
    - _index.json for quick lookup
    - Master layouts in _master_layouts/
    - Individual snapshot directories
    """
    
    # Parquet compression settings
    PARQUET_CONFIG = {
        "compression": "snappy",
    }
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize snapshot storage.
        
        Args:
            cache_dir: Root directory for snapshots (default from settings)
        """
        self.cache_dir = cache_dir or settings.SNAPSHOT_CACHE_DIR
        self.master_layouts_dir = self.cache_dir / settings.SNAPSHOT_MASTER_LAYOUTS_DIR
        self.index_file = self.cache_dir / settings.SNAPSHOT_INDEX_FILE
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.master_layouts_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Index Operations
    # =========================================================================
    
    def load_index(self) -> Dict[str, Any]:
        """
        Load _index.json, returning empty structure if missing.
        
        Returns:
            Index dictionary with version and snapshots by base_sql_file
        """
        if not self.index_file.exists():
            return {
                "version": 1,
                "last_updated": None,
                "snapshots": {}
            }
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[SNAPSHOT] Warning: Failed to load index, returning empty: {e}")
            return {
                "version": 1,
                "last_updated": None,
                "snapshots": {}
            }
    
    def save_index(self, index: Dict[str, Any]) -> None:
        """
        Save index to _index.json.
        
        Args:
            index: Index dictionary to save
        """
        index["last_updated"] = datetime.utcnow().isoformat() + "Z"
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, default=str)
    
    def update_index(self, snapshot_info: SnapshotInfo) -> None:
        """
        Add or update a snapshot entry in the index.
        
        Args:
            snapshot_info: Snapshot metadata to add/update
        """
        index = self.load_index()
        
        base_sql = snapshot_info.base_sql_file
        if base_sql not in index["snapshots"]:
            index["snapshots"][base_sql] = []
        
        # Create index entry
        entry = {
            "snapshot_id": snapshot_info.snapshot_id,
            "block_number": snapshot_info.block_number,
            "block_timestamp": snapshot_info.block_timestamp.isoformat() if snapshot_info.block_timestamp else None,
            "label": snapshot_info.label,
            "node_count": snapshot_info.node_count,
            "edge_count": snapshot_info.edge_count,
            "created_at": snapshot_info.created_at.isoformat() if snapshot_info.created_at else None
        }
        
        # Check if entry already exists (update)
        existing_idx = None
        for idx, existing in enumerate(index["snapshots"][base_sql]):
            if existing["snapshot_id"] == snapshot_info.snapshot_id:
                existing_idx = idx
                break
        
        if existing_idx is not None:
            index["snapshots"][base_sql][existing_idx] = entry
        else:
            index["snapshots"][base_sql].append(entry)
        
        # Sort by block number descending
        index["snapshots"][base_sql].sort(key=lambda x: x["block_number"], reverse=True)
        
        self.save_index(index)
    
    def remove_from_index(self, base_sql_file: str, block_number: int) -> None:
        """
        Remove a snapshot entry from the index.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number of snapshot to remove
        """
        index = self.load_index()
        
        if base_sql_file not in index["snapshots"]:
            return
        
        snapshot_id = f"{base_sql_file}_block_{block_number}"
        index["snapshots"][base_sql_file] = [
            entry for entry in index["snapshots"][base_sql_file]
            if entry["snapshot_id"] != snapshot_id
        ]
        
        self.save_index(index)
    
    # =========================================================================
    # Master Layout Operations
    # =========================================================================
    
    def get_master_layout_path(self, base_sql_file: str) -> Path:
        """
        Get path to master layout file for a SQL file.
        
        Args:
            base_sql_file: Base SQL file name
            
        Returns:
            Path to master layout parquet file
        """
        return self.master_layouts_dir / f"{base_sql_file}_master.parquet"
    
    def load_master_layout(self, base_sql_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load master layout for a SQL file.
        
        Args:
            base_sql_file: Base SQL file name
            
        Returns:
            Dictionary of {node_id: {x, y, first_seen}}
        """
        path = self.get_master_layout_path(base_sql_file)
        
        if not path.exists():
            return {}
        
        try:
            df = pd.read_parquet(path)
            layout = {}
            for _, row in df.iterrows():
                layout[str(row['node_id'])] = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'first_seen': str(row.get('first_seen', 'initial'))
                }
            return layout
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Failed to load master layout: {e}")
            return {}
    
    def save_master_layout(self, base_sql_file: str, layout: Dict[str, Dict[str, Any]]) -> None:
        """
        Save master layout to parquet file.
        
        Args:
            base_sql_file: Base SQL file name
            layout: Dictionary of {node_id: {x, y, first_seen}}
        """
        path = self.get_master_layout_path(base_sql_file)
        
        # Convert to DataFrame
        rows = []
        for node_id, pos in layout.items():
            rows.append({
                'node_id': str(node_id),
                'x': float(pos['x']),
                'y': float(pos['y']),
                'first_seen': str(pos.get('first_seen', 'initial'))
            })
        
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False, **self.PARQUET_CONFIG)
        
        print(f"[SNAPSHOT] Saved master layout: {len(layout)} nodes to {path.name}")
    
    def master_layout_exists(self, base_sql_file: str) -> bool:
        """Check if master layout exists for a SQL file."""
        return self.get_master_layout_path(base_sql_file).exists()
    
    def initialize_master_from_live(
        self, 
        base_sql_file: str, 
        live_layout: Dict[str, Dict[str, float]],
        source_id: str = "initial"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Initialize master layout from live layout.
        
        Args:
            base_sql_file: Base SQL file name
            live_layout: Live layout positions {node_id: {x, y}}
            source_id: Identifier for first_seen field
            
        Returns:
            Master layout dictionary with first_seen added
        """
        master = {}
        for node_id, pos in live_layout.items():
            master[str(node_id)] = {
                'x': float(pos['x']),
                'y': float(pos['y']),
                'first_seen': source_id
            }
        
        self.save_master_layout(base_sql_file, master)
        print(f"[SNAPSHOT] Initialized master layout from live: {len(master)} nodes")
        return master
    
    # =========================================================================
    # Snapshot Operations
    # =========================================================================
    
    def get_snapshot_dir(self, base_sql_file: str, block_number: int) -> Path:
        """
        Get path to snapshot directory.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            Path to snapshot directory
        """
        return self.cache_dir / base_sql_file / f"block_{block_number}"
    
    def get_snapshot_id(self, base_sql_file: str, block_number: int) -> str:
        """Generate snapshot ID from components."""
        return f"{base_sql_file}_block_{block_number}"
    
    def snapshot_exists(self, base_sql_file: str, block_number: int) -> bool:
        """
        Check if a snapshot exists and is valid.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            True if snapshot directory and metadata exist
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        metadata_path = snapshot_dir / "metadata.json"
        return snapshot_dir.exists() and metadata_path.exists()
    
    def save_snapshot(
        self,
        base_sql_file: str,
        block_number: int,
        edges_df: pd.DataFrame,
        layout: Dict[str, Dict[str, float]],
        metrics_df: Optional[pd.DataFrame],
        metadata: SnapshotMetadata
    ) -> None:
        """
        Save snapshot to disk.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            edges_df: DataFrame with source, target columns
            layout: Position dictionary {node_id: {x, y}}
            metrics_df: Optional metrics DataFrame
            metadata: Snapshot metadata
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        checksums = {}
        
        # Save edges
        edges_path = snapshot_dir / "edges.parquet"
        edges_df[['source', 'target']].to_parquet(
            edges_path, index=False, **self.PARQUET_CONFIG
        )
        checksums['edges'] = self._compute_checksum(edges_path)
        
        # Save layout
        layout_path = snapshot_dir / "layout.parquet"
        layout_df = pd.DataFrame([
            {'node_id': str(k), 'x': float(v['x']), 'y': float(v['y'])}
            for k, v in layout.items()
        ])
        layout_df.to_parquet(layout_path, index=False, **self.PARQUET_CONFIG)
        checksums['layout'] = self._compute_checksum(layout_path)
        
        # Save metrics (if provided)
        if metrics_df is not None and len(metrics_df) > 0:
            metrics_path = snapshot_dir / "metrics.parquet"
            metrics_df.to_parquet(metrics_path, index=False, **self.PARQUET_CONFIG)
            checksums['metrics'] = self._compute_checksum(metrics_path)
        
        # Update metadata with checksums
        metadata.checksums = checksums
        
        # Save metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(mode='json'), f, indent=2, default=str)
        
        print(f"[SNAPSHOT] Saved snapshot: {metadata.snapshot_id}")
    
    def load_snapshot(self, base_sql_file: str, block_number: int) -> SnapshotData:
        """
        Load full snapshot data.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            SnapshotData with edges, layout, metrics, and metadata
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        snapshot_id = self.get_snapshot_id(base_sql_file, block_number)
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        # Load metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        
        # Convert metadata to SnapshotInfo
        metadata = SnapshotInfo(
            snapshot_id=meta_dict['snapshot_id'],
            base_sql_file=meta_dict['base_sql_file'],
            block_number=meta_dict['block_number'],
            block_timestamp=datetime.fromisoformat(meta_dict['block_timestamp']) if meta_dict.get('block_timestamp') else None,
            label=meta_dict.get('label'),
            node_count=meta_dict['node_count'],
            edge_count=meta_dict['edge_count'],
            metrics_computed=meta_dict.get('metrics_computed', []),
            layout_source=LayoutSource(meta_dict.get('layout_source', 'master')),
            layout_unknown_nodes=meta_dict.get('layout_unknown_nodes', 0),
            created_at=datetime.fromisoformat(meta_dict['created_at']) if meta_dict.get('created_at') else datetime.utcnow(),
            status=SnapshotStatus.READY
        )
        
        # Load edges - vectorized conversion
        edges_path = snapshot_dir / "edges.parquet"
        edges_df = pd.read_parquet(edges_path)
        edges_df['source'] = edges_df['source'].astype(str)
        edges_df['target'] = edges_df['target'].astype(str)
        edges = edges_df[['source', 'target']].to_dict(orient='records')
        
        # Load layout - vectorized conversion
        layout_path = snapshot_dir / "layout.parquet"
        layout_df = pd.read_parquet(layout_path)
        layout_df['node_id'] = layout_df['node_id'].astype(str)
        layout_df['x'] = layout_df['x'].astype(float)
        layout_df['y'] = layout_df['y'].astype(float)
        layout = {
            row['node_id']: {'x': row['x'], 'y': row['y']}
            for row in layout_df[['node_id', 'x', 'y']].to_dict(orient='records')
        }
        
        # Load metrics (if exists) - vectorized conversion
        metrics = {}
        metrics_path = snapshot_dir / "metrics.parquet"
        if metrics_path.exists():
            try:
                metrics_df = pd.read_parquet(metrics_path)
                id_col = 'avatar' if 'avatar' in metrics_df.columns else metrics_df.columns[0]
                metrics_df[id_col] = metrics_df[id_col].astype(str)
                
                # Replace NaN/Inf with None for JSON serialization
                metrics_df = metrics_df.replace([float('inf'), float('-inf')], None)
                metrics_df = metrics_df.where(pd.notnull(metrics_df), None)
                
                # Convert to dict efficiently
                metric_cols = [col for col in metrics_df.columns if col != id_col]
                for record in metrics_df.to_dict(orient='records'):
                    node_id = record[id_col]
                    metrics[node_id] = {
                        col: self._safe_value(record[col])
                        for col in metric_cols
                    }
            except Exception as e:
                print(f"[SNAPSHOT] Warning: Failed to load metrics: {e}")
        
        return SnapshotData(
            snapshot_id=snapshot_id,
            edges=edges,
            layout=layout,
            metrics=metrics,
            metadata=metadata
        )
    
    def load_snapshot_metadata(self, base_sql_file: str, block_number: int) -> Optional[SnapshotInfo]:
        """
        Load only snapshot metadata (fast operation).
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            SnapshotInfo or None if not found
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        metadata_path = snapshot_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
            
            return SnapshotInfo(
                snapshot_id=meta_dict['snapshot_id'],
                base_sql_file=meta_dict['base_sql_file'],
                block_number=meta_dict['block_number'],
                block_timestamp=datetime.fromisoformat(meta_dict['block_timestamp']) if meta_dict.get('block_timestamp') else None,
                label=meta_dict.get('label'),
                node_count=meta_dict['node_count'],
                edge_count=meta_dict['edge_count'],
                metrics_computed=meta_dict.get('metrics_computed', []),
                layout_source=LayoutSource(meta_dict.get('layout_source', 'master')),
                layout_unknown_nodes=meta_dict.get('layout_unknown_nodes', 0),
                created_at=datetime.fromisoformat(meta_dict['created_at']) if meta_dict.get('created_at') else datetime.utcnow(),
                status=SnapshotStatus.READY
            )
        except Exception as e:
            print(f"[SNAPSHOT] Warning: Failed to load metadata: {e}")
            return None

    def load_snapshot_nodes(self, base_sql_file: str, block_number: int) -> Dict[str, Any]:
        """
        Load snapshot nodes with positions and metrics (no edges).
        
        Returns Cytoscape-compatible node elements for fast initial render.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            Dict with 'elements' list of Cytoscape node objects
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        # Load layout - vectorized
        layout_path = snapshot_dir / "layout.parquet"
        layout_df = pd.read_parquet(layout_path)
        layout_df['node_id'] = layout_df['node_id'].astype(str)
        
        # Load metrics if exists
        metrics_dict = {}
        metrics_path = snapshot_dir / "metrics.parquet"
        if metrics_path.exists():
            try:
                metrics_df = pd.read_parquet(metrics_path)
                id_col = 'avatar' if 'avatar' in metrics_df.columns else metrics_df.columns[0]
                metrics_df[id_col] = metrics_df[id_col].astype(str)
                
                # Replace NaN/Inf
                metrics_df = metrics_df.replace([float('inf'), float('-inf')], None)
                metrics_df = metrics_df.where(pd.notnull(metrics_df), None)
                
                metric_cols = [col for col in metrics_df.columns if col != id_col]
                for record in metrics_df.to_dict(orient='records'):
                    node_id = record[id_col]
                    metrics_dict[node_id] = {
                        col: self._safe_value(record[col])
                        for col in metric_cols
                    }
            except Exception as e:
                print(f"[SNAPSHOT] Warning: Failed to load metrics for nodes: {e}")
        
        # Build Cytoscape elements
        elements = []
        for _, row in layout_df.iterrows():
            node_id = row['node_id']
            node_metrics = metrics_dict.get(node_id, {})
            
            elements.append({
                "group": "nodes",
                "data": {
                    "id": node_id,
                    **node_metrics
                },
                "position": {
                    "x": float(row['x']),
                    "y": float(row['y'])
                }
            })
        
        return {"elements": elements}
    
    def load_snapshot_edges(
        self, 
        base_sql_file: str, 
        block_number: int,
        offset: int = 0,
        limit: int = 50000
    ) -> Dict[str, Any]:
        """
        Load snapshot edges with pagination.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            offset: Starting offset
            limit: Max edges to return
            
        Returns:
            Dict with 'edges' list, 'total', 'has_more'
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        edges_path = snapshot_dir / "edges.parquet"
        edges_df = pd.read_parquet(edges_path)
        
        total = len(edges_df)
        
        # Slice for pagination - use .copy() to avoid SettingWithCopyWarning
        end_idx = min(offset + limit, total)
        edges_slice = edges_df.iloc[offset:end_idx].copy()
        
        # Convert to string
        edges_slice['source'] = edges_slice['source'].astype(str)
        edges_slice['target'] = edges_slice['target'].astype(str)
        
        # Convert to Cytoscape format efficiently using vectorized operations
        edges = [
            {
                "group": "edges",
                "data": {
                    "id": f"{row['source']}->{row['target']}",
                    "source": row['source'],
                    "target": row['target']
                }
            }
            for row in edges_slice.to_dict(orient='records')
        ]
        
        return {
            "edges": edges,
            "total": total,
            "offset": offset,
            "has_more": end_idx < total
        }
    
    def load_snapshot_edges_lightweight(
        self,
        base_sql_file: str,
        block_number: int
    ) -> List[List[str]]:
        """
        Load ALL snapshot edges in lightweight format for animation.
        
        Returns just [source, target] pairs - no Cytoscape wrapping.
        Much faster and uses less memory than full edge loading.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            List of [source, target] pairs
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        
        if not snapshot_dir.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        edges_path = snapshot_dir / "edges.parquet"
        
        # Load only source/target columns - much faster
        edges_df = pd.read_parquet(edges_path, columns=['source', 'target'])
        
        # Convert to strings
        edges_df['source'] = edges_df['source'].astype(str)
        edges_df['target'] = edges_df['target'].astype(str)
        
        # Return as list of [source, target] pairs - very compact
        return edges_df.values.tolist()
    
    def delete_snapshot(self, base_sql_file: str, block_number: int) -> bool:
        """
        Delete a snapshot and update index.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            True if deleted, False if not found
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        
        if not snapshot_dir.exists():
            return False
        
        try:
            shutil.rmtree(snapshot_dir)
            self.remove_from_index(base_sql_file, block_number)
            print(f"[SNAPSHOT] Deleted snapshot: {base_sql_file}_block_{block_number}")
            return True
        except Exception as e:
            print(f"[SNAPSHOT] Error deleting snapshot: {e}")
            return False
    
    def list_snapshots(self, base_sql_file: Optional[str] = None) -> List[SnapshotInfo]:
        """
        List available snapshots from index.
        
        Args:
            base_sql_file: Optional filter by SQL file
            
        Returns:
            List of SnapshotInfo objects
        """
        index = self.load_index()
        snapshots = []
        
        sql_files = [base_sql_file] if base_sql_file else index["snapshots"].keys()
        
        for sql_file in sql_files:
            if sql_file not in index["snapshots"]:
                continue
            
            for entry in index["snapshots"][sql_file]:
                # Load full metadata if needed, or construct from index
                info = SnapshotInfo(
                    snapshot_id=entry['snapshot_id'],
                    base_sql_file=sql_file,
                    block_number=entry['block_number'],
                    block_timestamp=datetime.fromisoformat(entry['block_timestamp']) if entry.get('block_timestamp') else None,
                    label=entry.get('label'),
                    node_count=entry.get('node_count', 0),
                    edge_count=entry.get('edge_count', 0),
                    metrics_computed=[],  # Not stored in index
                    layout_source=LayoutSource.MASTER,  # Not stored in index
                    layout_unknown_nodes=0,
                    created_at=datetime.fromisoformat(entry['created_at']) if entry.get('created_at') else datetime.utcnow(),
                    status=SnapshotStatus.READY
                )
                snapshots.append(info)
        
        return snapshots
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum for a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"
    
    def _safe_value(self, value: Any) -> Any:
        """Convert value to JSON-safe format."""
        import numpy as np
        
        if value is None:
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage usage statistics."""
        total_size = 0
        snapshot_count = 0
        snapshots_by_sql = {}
        master_layout_count = 0
        
        # Count master layouts
        for f in self.master_layouts_dir.glob("*.parquet"):
            master_layout_count += 1
            total_size += f.stat().st_size
        
        # Count snapshots
        index = self.load_index()
        for sql_file, entries in index.get("snapshots", {}).items():
            count = len(entries)
            snapshots_by_sql[sql_file] = count
            snapshot_count += count
            
            # Sum sizes
            sql_dir = self.cache_dir / sql_file
            if sql_dir.exists():
                for f in sql_dir.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
        
        return StorageStats(
            total_size_bytes=total_size,
            snapshot_count=snapshot_count,
            snapshots_by_sql_file=snapshots_by_sql,
            master_layout_count=master_layout_count
        )
    
    def get_available_snapshot_sql_files(self) -> List[str]:
        """
        Get list of available snapshot SQL template files.
        
        Returns:
            List of base SQL file names that have snapshot templates
        """
        available = []
        if settings.SNAPSHOT_SQL_DIR.exists():
            for sql_path in settings.SNAPSHOT_SQL_DIR.glob("*_snapshot.sql"):
                # Extract base name: crc_v2_trusts_snapshot.sql -> crc_v2_trusts
                base_name = sql_path.stem.replace("_snapshot", "")
                available.append(base_name)
        return available
    
    # =========================================================================
    # Comparison Helper Methods
    # =========================================================================
    
    def load_snapshot_node_ids(self, base_sql_file: str, block_number: int) -> Set[str]:
        """
        Load only node IDs from a snapshot (fast).
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            Set of node IDs
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        layout_path = snapshot_dir / "layout.parquet"
        
        if not layout_path.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        layout_df = pd.read_parquet(layout_path, columns=['node_id'])
        return set(layout_df['node_id'].astype(str).tolist())
    
    def load_snapshot_edge_set(self, base_sql_file: str, block_number: int) -> Set[Tuple[str, str]]:
        """
        Load edges as a set of (source, target) tuples.
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            Set of (source, target) tuples
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        edges_path = snapshot_dir / "edges.parquet"
        
        if not edges_path.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        edges_df = pd.read_parquet(edges_path)
        edges_df['source'] = edges_df['source'].astype(str)
        edges_df['target'] = edges_df['target'].astype(str)
        
        return set(zip(edges_df['source'].tolist(), edges_df['target'].tolist()))
    
    def load_snapshot_layout_dict(self, base_sql_file: str, block_number: int) -> Dict[str, Dict[str, float]]:
        """
        Load layout as a dictionary (for comparison/animation).
        
        Args:
            base_sql_file: Base SQL file name
            block_number: Block number
            
        Returns:
            Dict mapping node_id -> {x, y}
        """
        snapshot_dir = self.get_snapshot_dir(base_sql_file, block_number)
        layout_path = snapshot_dir / "layout.parquet"
        
        if not layout_path.exists():
            raise ValueError(f"Snapshot not found: {base_sql_file}_block_{block_number}")
        
        layout_df = pd.read_parquet(layout_path)
        layout_df['node_id'] = layout_df['node_id'].astype(str)
        
        return {
            row['node_id']: {'x': float(row['x']), 'y': float(row['y'])}
            for row in layout_df.to_dict(orient='records')
        }