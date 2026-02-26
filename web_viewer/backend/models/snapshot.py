"""
Snapshot Models

Pydantic models for historical network snapshots.
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class SnapshotStatus(str, Enum):
    """Status of a snapshot."""
    PENDING = "pending"
    COMPUTING = "computing"
    READY = "ready"
    ERROR = "error"


class MetricsMode(str, Enum):
    """Metrics computation mode for snapshots."""
    NONE = "none"           # No metrics computed
    BASIC = "basic"         # Degree metrics only
    STANDARD = "standard"   # Common metrics (degree, pagerank, betweenness)
    FULL = "full"           # All available metrics


class LayoutSource(str, Enum):
    """Source of layout positions for a snapshot."""
    MASTER = "master"           # All positions from master layout
    COMPUTED = "computed"       # Some positions computed via spring


# =============================================================================
# Request Models
# =============================================================================

class SnapshotCreateRequest(BaseModel):
    """Request to create a single snapshot."""
    base_sql_file: str = Field(
        description="Base SQL file name (without extension), e.g., 'crc_v2_trusts'"
    )
    block_number: int = Field(
        description="Target block number for the snapshot"
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label for the snapshot"
    )
    metrics_mode: MetricsMode = Field(
        default=MetricsMode.STANDARD,
        description="Which metrics to compute for the snapshot"
    )


class SnapshotBatchRequest(BaseModel):
    """Request to create multiple snapshots."""
    base_sql_file: str = Field(
        description="Base SQL file name (without extension)"
    )
    block_numbers: List[int] = Field(
        description="List of block numbers to create snapshots for"
    )
    metrics_mode: MetricsMode = Field(
        default=MetricsMode.STANDARD,
        description="Which metrics to compute"
    )


class SnapshotSuggestRequest(BaseModel):
    """Request for suggested block numbers."""
    base_sql_file: str = Field(
        description="Base SQL file name"
    )
    interval: str = Field(
        default="daily",
        description="Interval for suggestions: 'daily', 'weekly', 'monthly'"
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Start date for range (defaults to 30 days ago)"
    )
    end_date: Optional[date] = Field(
        default=None,
        description="End date for range (defaults to today)"
    )
    count: Optional[int] = Field(
        default=30,
        description="Maximum number of suggestions to return"
    )


class SnapshotRebuildRequest(BaseModel):
    """Request to rebuild snapshots from scratch."""
    base_sql_file: str = Field(
        description="Base SQL file name (e.g., 'crc_v2_trusts')"
    )
    block_numbers: Optional[List[int]] = Field(
        default=None,
        description="Specific blocks to rebuild. If None, rebuild all existing."
    )
    metrics_mode: MetricsMode = Field(
        default=MetricsMode.STANDARD,
        description="Which metrics to compute"
    )
    delete_existing: bool = Field(
        default=True,
        description="Delete existing snapshots before rebuilding"
    )


# =============================================================================
# Response Models
# =============================================================================

class SnapshotInfo(BaseModel):
    """Information about a snapshot."""
    snapshot_id: str = Field(
        description="Unique identifier: '{base_sql_file}_block_{number}'"
    )
    base_sql_file: str = Field(
        description="Base SQL file name"
    )
    block_number: int = Field(
        description="Block number"
    )
    block_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the block"
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label"
    )
    node_count: int = Field(
        description="Number of nodes in snapshot"
    )
    edge_count: int = Field(
        description="Number of edges in snapshot"
    )
    metrics_computed: List[str] = Field(
        default_factory=list,
        description="List of computed metric names"
    )
    layout_source: LayoutSource = Field(
        description="Where positions came from"
    )
    layout_unknown_nodes: int = Field(
        default=0,
        description="Number of nodes positioned via spring algorithm"
    )
    created_at: datetime = Field(
        description="When the snapshot was created"
    )
    status: SnapshotStatus = Field(
        description="Current status of the snapshot"
    )


class SnapshotListResponse(BaseModel):
    """Response containing list of snapshots."""
    snapshots: List[SnapshotInfo] = Field(
        description="List of snapshot info objects"
    )
    total_count: int = Field(
        description="Total number of snapshots"
    )


class SnapshotData(BaseModel):
    """Full snapshot data including edges, layout, and metrics."""
    snapshot_id: str = Field(
        description="Unique identifier"
    )
    edges: List[Dict[str, str]] = Field(
        description="List of edges as {source, target} dicts"
    )
    layout: Dict[str, Dict[str, float]] = Field(
        description="Node positions as {node_id: {x, y}}"
    )
    metrics: Dict[str, Dict[str, Any]] = Field(
        description="Node metrics as {node_id: {metric: value}}"
    )
    metadata: SnapshotInfo = Field(
        description="Snapshot metadata"
    )


class BlockSuggestion(BaseModel):
    """A suggested block number for snapshot creation."""
    block_number: int = Field(
        description="Block number"
    )
    timestamp: datetime = Field(
        description="Block timestamp"
    )
    label: str = Field(
        description="Human-readable label, e.g., '2024-01-15 (Monday)'"
    )


class SnapshotSuggestResponse(BaseModel):
    """Response containing suggested block numbers."""
    suggestions: List[BlockSuggestion] = Field(
        description="List of block suggestions"
    )


class SnapshotProgress(BaseModel):
    """Progress update for snapshot creation."""
    snapshot_id: str = Field(
        description="Snapshot being created"
    )
    status: SnapshotStatus = Field(
        description="Current status"
    )
    stage: str = Field(
        description="Current stage: 'sql', 'layout', 'metrics', 'saving'"
    )
    progress_percent: int = Field(
        ge=0, le=100,
        description="Progress percentage"
    )
    message: str = Field(
        description="Human-readable status message"
    )


class StorageStats(BaseModel):
    """Statistics about snapshot storage."""
    total_size_bytes: int = Field(
        description="Total storage size in bytes"
    )
    snapshot_count: int = Field(
        description="Total number of snapshots"
    )
    snapshots_by_sql_file: Dict[str, int] = Field(
        description="Count of snapshots per SQL file"
    )
    master_layout_count: int = Field(
        description="Number of master layout files"
    )


# =============================================================================
# Internal Models (for service layer)
# =============================================================================

class SnapshotMetadata(BaseModel):
    """Full metadata stored in metadata.json for a snapshot."""
    snapshot_id: str
    base_sql_file: str
    block_number: int
    block_timestamp: Optional[datetime] = None
    label: Optional[str] = None
    node_count: int
    edge_count: int
    metrics_computed: List[str] = []
    metrics_mode: str = "standard"
    layout_source: str = "master"
    layout_unknown_nodes: int = 0
    created_at: datetime
    computation_time_seconds: float = 0.0
    files: Dict[str, str] = Field(
        default_factory=lambda: {
            "edges": "edges.parquet",
            "layout": "layout.parquet", 
            "metrics": "metrics.parquet"
        }
    )
    checksums: Dict[str, str] = Field(
        default_factory=dict,
        description="SHA256 checksums for files"
    )


class MasterLayoutEntry(BaseModel):
    """Entry in master layout tracking when/where node was first positioned."""
    node_id: str
    x: float
    y: float
    first_seen: str = Field(
        description="Snapshot ID where this node was first positioned"
    )


class IndexEntry(BaseModel):
    """Entry in the _index.json file for quick lookup."""
    snapshot_id: str
    block_number: int
    block_timestamp: Optional[datetime] = None
    label: Optional[str] = None
    node_count: int
    edge_count: int
    created_at: datetime