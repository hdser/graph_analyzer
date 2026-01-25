"""
Snapshot Diff Models

Pydantic models for diff-based snapshot storage.
Enables 70-90% storage reduction by storing incremental changes
instead of full state for each snapshot.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from pydantic import BaseModel, Field


class SnapshotDiff(BaseModel):
    """
    Represents the difference between two snapshots.

    Used to store incremental changes instead of full state.
    Anchor snapshots store full state, diff snapshots store changes.
    """
    snapshot_id: str = Field(
        description="Unique identifier for this snapshot"
    )
    base_snapshot_id: Optional[str] = Field(
        default=None,
        description="ID of the base snapshot this diff is relative to. None for anchor snapshots."
    )
    is_anchor: bool = Field(
        default=False,
        description="If True, this is a full snapshot (not a diff)"
    )

    # Node changes
    added_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs added since base snapshot"
    )
    removed_nodes: List[str] = Field(
        default_factory=list,
        description="Node IDs removed since base snapshot"
    )

    # Edge changes (stored as (source, target) tuples)
    added_edges: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Edges added since base snapshot as (source, target)"
    )
    removed_edges: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="Edges removed since base snapshot as (source, target)"
    )

    # Positions for new nodes only
    new_positions: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Positions for added nodes as {node_id: {x, y}}"
    )

    # Metrics for new/changed nodes
    new_metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metrics for added nodes as {node_id: {metric: value}}"
    )

    # Metadata
    block_number: int = Field(
        description="Block number of this snapshot"
    )
    block_timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp of the block"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this diff was created"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DiffChainInfo(BaseModel):
    """
    Information about a chain of diffs leading to a snapshot.

    Used to determine if reconstruction is feasible or if
    a new anchor should be created.
    """
    snapshot_id: str = Field(
        description="Target snapshot ID"
    )
    anchor_id: str = Field(
        description="ID of the anchor snapshot this chain starts from"
    )
    chain_length: int = Field(
        description="Number of diffs in the chain (0 for anchor)"
    )
    diff_ids: List[str] = Field(
        default_factory=list,
        description="Ordered list of diff snapshot IDs from anchor to target"
    )
    total_changes: int = Field(
        default=0,
        description="Total number of changes across all diffs in chain"
    )


class ReconstructedSnapshot(BaseModel):
    """
    A snapshot reconstructed from a chain of diffs.

    Contains the full state computed by applying diffs to an anchor.
    """
    snapshot_id: str = Field(
        description="Snapshot ID"
    )
    nodes: List[str] = Field(
        description="All node IDs in the reconstructed snapshot"
    )
    edges: List[Tuple[str, str]] = Field(
        description="All edges as (source, target) tuples"
    )
    positions: Dict[str, Dict[str, float]] = Field(
        description="All node positions as {node_id: {x, y}}"
    )
    metrics: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="All node metrics"
    )

    # Reconstruction metadata
    anchor_id: str = Field(
        description="ID of anchor snapshot used for reconstruction"
    )
    chain_length: int = Field(
        description="Number of diffs applied"
    )
    reconstruction_time_ms: float = Field(
        default=0.0,
        description="Time taken to reconstruct in milliseconds"
    )


class DiffMetadata(BaseModel):
    """
    Metadata stored alongside a diff snapshot.

    Stored in diff_metadata.json within the snapshot directory.
    """
    snapshot_id: str
    base_snapshot_id: Optional[str] = None
    is_anchor: bool = False

    # Statistics
    added_node_count: int = 0
    removed_node_count: int = 0
    added_edge_count: int = 0
    removed_edge_count: int = 0

    # Block info
    block_number: int
    block_timestamp: Optional[datetime] = None

    # Chain info
    chain_anchor_id: Optional[str] = None
    chain_position: int = 0  # 0 for anchor, 1+ for diffs

    # Storage info
    files: Dict[str, str] = Field(
        default_factory=lambda: {
            "diff": "diff.parquet",
            "new_positions": "new_positions.parquet"
        }
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class DiffComputeResult(BaseModel):
    """
    Result of computing a diff between two snapshots.
    """
    diff: SnapshotDiff
    storage_savings_percent: float = Field(
        description="Estimated storage savings compared to full snapshot"
    )
    should_be_anchor: bool = Field(
        default=False,
        description="Whether this should be stored as an anchor instead"
    )
    computation_time_ms: float = 0.0


class AnchorDecision(BaseModel):
    """
    Decision about whether to create an anchor snapshot.
    """
    create_anchor: bool = Field(
        description="Whether to create a new anchor"
    )
    reason: str = Field(
        description="Reason for the decision"
    )
    chain_length: int = Field(
        default=0,
        description="Current chain length if not creating anchor"
    )
    total_changes: int = Field(
        default=0,
        description="Total changes in current chain"
    )
