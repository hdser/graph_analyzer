"""
Diff Storage Service

Handles diff-based snapshot storage for 70-90% storage reduction.
Stores incremental changes instead of full state for most snapshots.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..config import settings
from ..models.snapshot_diff import (
    SnapshotDiff,
    DiffChainInfo,
    ReconstructedSnapshot,
    DiffMetadata,
    DiffComputeResult,
    AnchorDecision,
)


class DiffStorageService:
    """
    Service for diff-based snapshot storage.

    Features:
    - Compute diffs between snapshots
    - Store diffs efficiently using Parquet
    - Reconstruct snapshots from diff chains
    - Decide when to create anchors vs diffs
    """

    # Maximum chain length before forcing a new anchor
    MAX_CHAIN_LENGTH = 10

    # Maximum total changes in a chain before forcing anchor
    MAX_CHAIN_CHANGES = 50000

    # Minimum storage savings to use diff (vs full snapshot)
    MIN_SAVINGS_PERCENT = 30

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize diff storage service.

        Args:
            cache_dir: Root directory for snapshots
        """
        self.cache_dir = cache_dir or settings.SNAPSHOT_CACHE_DIR
        self._diff_index: Dict[str, Dict] = {}  # base_sql -> {snapshot_id -> DiffMetadata}

    # =========================================================================
    # Diff Computation
    # =========================================================================

    def compute_diff(
        self,
        base_sql_file: str,
        from_nodes: Set[str],
        from_edges: Set[Tuple[str, str]],
        to_nodes: Set[str],
        to_edges: Set[Tuple[str, str]],
        to_positions: Dict[str, Dict[str, float]],
        to_metrics: Dict[str, Dict[str, Any]],
        from_block: int,
        to_block: int,
    ) -> DiffComputeResult:
        """
        Compute the diff between two snapshot states.

        Args:
            base_sql_file: Base SQL file name
            from_nodes: Node IDs in base snapshot
            from_edges: Edges in base snapshot as (source, target) tuples
            to_nodes: Node IDs in target snapshot
            to_edges: Edges in target snapshot
            to_positions: Positions for target snapshot
            to_metrics: Metrics for target snapshot
            from_block: Base snapshot block number
            to_block: Target snapshot block number

        Returns:
            DiffComputeResult with the computed diff and metadata
        """
        start_time = time.time()

        # Compute node changes
        added_nodes = list(to_nodes - from_nodes)
        removed_nodes = list(from_nodes - to_nodes)

        # Compute edge changes
        added_edges = list(to_edges - from_edges)
        removed_edges = list(from_edges - to_edges)

        # Get positions only for new nodes
        new_positions = {
            node_id: to_positions[node_id]
            for node_id in added_nodes
            if node_id in to_positions
        }

        # Get metrics only for new nodes
        new_metrics = {
            node_id: to_metrics.get(node_id, {})
            for node_id in added_nodes
        }

        snapshot_id = f"{base_sql_file}_block_{to_block}"
        base_snapshot_id = f"{base_sql_file}_block_{from_block}"

        diff = SnapshotDiff(
            snapshot_id=snapshot_id,
            base_snapshot_id=base_snapshot_id,
            is_anchor=False,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            new_positions=new_positions,
            new_metrics=new_metrics,
            block_number=to_block,
        )

        # Estimate storage savings
        total_changes = (
            len(added_nodes) + len(removed_nodes) +
            len(added_edges) + len(removed_edges)
        )
        total_full = len(to_nodes) + len(to_edges)

        if total_full > 0:
            savings_percent = (1 - (total_changes / total_full)) * 100
        else:
            savings_percent = 0

        # Decide if this should be an anchor
        should_be_anchor = savings_percent < self.MIN_SAVINGS_PERCENT

        computation_time = (time.time() - start_time) * 1000

        return DiffComputeResult(
            diff=diff,
            storage_savings_percent=savings_percent,
            should_be_anchor=should_be_anchor,
            computation_time_ms=computation_time,
        )

    # =========================================================================
    # Anchor Decisions
    # =========================================================================

    def should_create_anchor(
        self,
        base_sql_file: str,
        block_number: int,
    ) -> AnchorDecision:
        """
        Determine if a new anchor should be created for this snapshot.

        Args:
            base_sql_file: Base SQL file name
            block_number: Target block number

        Returns:
            AnchorDecision with the decision and reasoning
        """
        # Get diff index for this base file
        diff_index = self._load_diff_index(base_sql_file)

        if not diff_index:
            # No existing snapshots, this must be an anchor
            return AnchorDecision(
                create_anchor=True,
                reason="First snapshot for this SQL file",
                chain_length=0,
                total_changes=0,
            )

        # Check if this is configured interval for anchors
        anchor_interval = getattr(settings, 'SNAPSHOT_ANCHOR_INTERVAL', 10)

        # Find the most recent anchor
        anchors = [
            entry for entry in diff_index.values()
            if entry.get("is_anchor", False)
        ]

        if not anchors:
            # No anchors exist, create one
            return AnchorDecision(
                create_anchor=True,
                reason="No existing anchors",
                chain_length=0,
                total_changes=0,
            )

        # Sort by block number to find closest anchor
        anchors.sort(key=lambda x: x["block_number"], reverse=True)
        latest_anchor = anchors[0]

        # Count snapshots since last anchor
        snapshots_since_anchor = sum(
            1 for entry in diff_index.values()
            if entry["block_number"] > latest_anchor["block_number"]
        )

        # Check if we've exceeded chain length
        if snapshots_since_anchor >= self.MAX_CHAIN_LENGTH:
            return AnchorDecision(
                create_anchor=True,
                reason=f"Chain length ({snapshots_since_anchor}) exceeds maximum ({self.MAX_CHAIN_LENGTH})",
                chain_length=snapshots_since_anchor,
                total_changes=0,
            )

        # Check interval-based anchoring
        if anchor_interval > 0:
            block_diff = block_number - latest_anchor["block_number"]
            # Simple heuristic: if block numbers differ significantly
            if snapshots_since_anchor >= anchor_interval:
                return AnchorDecision(
                    create_anchor=True,
                    reason=f"Anchor interval ({anchor_interval}) reached",
                    chain_length=snapshots_since_anchor,
                    total_changes=0,
                )

        return AnchorDecision(
            create_anchor=False,
            reason="Within acceptable chain parameters",
            chain_length=snapshots_since_anchor,
            total_changes=0,
        )

    # =========================================================================
    # Storage Operations
    # =========================================================================

    def save_as_diff(
        self,
        base_sql_file: str,
        diff: SnapshotDiff,
    ) -> Path:
        """
        Save a diff snapshot to storage.

        Args:
            base_sql_file: Base SQL file name
            diff: The diff to save

        Returns:
            Path to the snapshot directory
        """
        snapshot_dir = self._get_snapshot_dir(base_sql_file, diff.block_number)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Save diff data as parquet
        diff_data = {
            "type": [],
            "node_id": [],
            "source": [],
            "target": [],
        }

        # Added nodes
        for node_id in diff.added_nodes:
            diff_data["type"].append("added_node")
            diff_data["node_id"].append(node_id)
            diff_data["source"].append(None)
            diff_data["target"].append(None)

        # Removed nodes
        for node_id in diff.removed_nodes:
            diff_data["type"].append("removed_node")
            diff_data["node_id"].append(node_id)
            diff_data["source"].append(None)
            diff_data["target"].append(None)

        # Added edges
        for source, target in diff.added_edges:
            diff_data["type"].append("added_edge")
            diff_data["node_id"].append(None)
            diff_data["source"].append(source)
            diff_data["target"].append(target)

        # Removed edges
        for source, target in diff.removed_edges:
            diff_data["type"].append("removed_edge")
            diff_data["node_id"].append(None)
            diff_data["source"].append(source)
            diff_data["target"].append(target)

        df = pd.DataFrame(diff_data)
        diff_path = snapshot_dir / "diff.parquet"
        df.to_parquet(diff_path, compression="snappy")

        # Save new positions
        if diff.new_positions:
            pos_data = {
                "node_id": [],
                "x": [],
                "y": [],
            }
            for node_id, pos in diff.new_positions.items():
                pos_data["node_id"].append(node_id)
                pos_data["x"].append(pos["x"])
                pos_data["y"].append(pos["y"])

            pos_df = pd.DataFrame(pos_data)
            pos_path = snapshot_dir / "new_positions.parquet"
            pos_df.to_parquet(pos_path, compression="snappy")

        # Save new metrics if present
        if diff.new_metrics:
            metrics_records = []
            for node_id, metrics in diff.new_metrics.items():
                record = {"node_id": node_id, **metrics}
                metrics_records.append(record)

            if metrics_records:
                metrics_df = pd.DataFrame(metrics_records)
                metrics_path = snapshot_dir / "new_metrics.parquet"
                metrics_df.to_parquet(metrics_path, compression="snappy")

        # Save diff metadata
        metadata = DiffMetadata(
            snapshot_id=diff.snapshot_id,
            base_snapshot_id=diff.base_snapshot_id,
            is_anchor=False,
            added_node_count=len(diff.added_nodes),
            removed_node_count=len(diff.removed_nodes),
            added_edge_count=len(diff.added_edges),
            removed_edge_count=len(diff.removed_edges),
            block_number=diff.block_number,
            block_timestamp=diff.block_timestamp,
        )

        metadata_path = snapshot_dir / "diff_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(), f, indent=2, default=str)

        # Update diff index
        self._update_diff_index(base_sql_file, metadata)

        return snapshot_dir

    def load_diff(
        self,
        base_sql_file: str,
        block_number: int,
    ) -> Optional[SnapshotDiff]:
        """
        Load a diff from storage.

        Args:
            base_sql_file: Base SQL file name
            block_number: Block number

        Returns:
            SnapshotDiff or None if not found
        """
        snapshot_dir = self._get_snapshot_dir(base_sql_file, block_number)
        diff_path = snapshot_dir / "diff.parquet"
        metadata_path = snapshot_dir / "diff_metadata.json"

        if not diff_path.exists() or not metadata_path.exists():
            return None

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Load diff data
        df = pd.read_parquet(diff_path)

        added_nodes = df[df["type"] == "added_node"]["node_id"].tolist()
        removed_nodes = df[df["type"] == "removed_node"]["node_id"].tolist()

        added_edges_df = df[df["type"] == "added_edge"]
        added_edges = list(zip(
            added_edges_df["source"].tolist(),
            added_edges_df["target"].tolist()
        ))

        removed_edges_df = df[df["type"] == "removed_edge"]
        removed_edges = list(zip(
            removed_edges_df["source"].tolist(),
            removed_edges_df["target"].tolist()
        ))

        # Load positions
        new_positions = {}
        pos_path = snapshot_dir / "new_positions.parquet"
        if pos_path.exists():
            pos_df = pd.read_parquet(pos_path)
            for _, row in pos_df.iterrows():
                new_positions[row["node_id"]] = {"x": row["x"], "y": row["y"]}

        # Load metrics
        new_metrics = {}
        metrics_path = snapshot_dir / "new_metrics.parquet"
        if metrics_path.exists():
            metrics_df = pd.read_parquet(metrics_path)
            for _, row in metrics_df.iterrows():
                node_id = row["node_id"]
                new_metrics[node_id] = {
                    k: v for k, v in row.items()
                    if k != "node_id" and pd.notna(v)
                }

        return SnapshotDiff(
            snapshot_id=metadata["snapshot_id"],
            base_snapshot_id=metadata.get("base_snapshot_id"),
            is_anchor=metadata.get("is_anchor", False),
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            added_edges=added_edges,
            removed_edges=removed_edges,
            new_positions=new_positions,
            new_metrics=new_metrics,
            block_number=block_number,
        )

    # =========================================================================
    # Reconstruction
    # =========================================================================

    def get_diff_chain(
        self,
        base_sql_file: str,
        block_number: int,
    ) -> Optional[DiffChainInfo]:
        """
        Get the chain of diffs needed to reconstruct a snapshot.

        Args:
            base_sql_file: Base SQL file name
            block_number: Target block number

        Returns:
            DiffChainInfo or None if not found
        """
        diff_index = self._load_diff_index(base_sql_file)
        snapshot_id = f"{base_sql_file}_block_{block_number}"

        if snapshot_id not in diff_index:
            return None

        entry = diff_index[snapshot_id]

        if entry.get("is_anchor", False):
            return DiffChainInfo(
                snapshot_id=snapshot_id,
                anchor_id=snapshot_id,
                chain_length=0,
                diff_ids=[],
            )

        # Walk back to find anchor
        chain = []
        current_id = snapshot_id
        total_changes = 0

        while current_id in diff_index:
            current = diff_index[current_id]

            if current.get("is_anchor", False):
                # Found anchor
                return DiffChainInfo(
                    snapshot_id=snapshot_id,
                    anchor_id=current_id,
                    chain_length=len(chain),
                    diff_ids=list(reversed(chain)),
                    total_changes=total_changes,
                )

            chain.append(current_id)
            total_changes += (
                current.get("added_node_count", 0) +
                current.get("removed_node_count", 0) +
                current.get("added_edge_count", 0) +
                current.get("removed_edge_count", 0)
            )

            base_id = current.get("base_snapshot_id")
            if not base_id or base_id == current_id:
                break  # Prevent infinite loop
            current_id = base_id

        return None

    def reconstruct_snapshot(
        self,
        base_sql_file: str,
        block_number: int,
        anchor_data: Dict[str, Any],
    ) -> Optional[ReconstructedSnapshot]:
        """
        Reconstruct a snapshot by applying diffs to an anchor.

        Args:
            base_sql_file: Base SQL file name
            block_number: Target block number
            anchor_data: Dict with 'nodes', 'edges', 'positions', 'metrics' from anchor

        Returns:
            ReconstructedSnapshot or None if reconstruction fails
        """
        start_time = time.time()

        chain = self.get_diff_chain(base_sql_file, block_number)
        if not chain:
            return None

        snapshot_id = f"{base_sql_file}_block_{block_number}"

        # Start with anchor data
        nodes = set(anchor_data.get("nodes", []))
        edges = set(tuple(e) if isinstance(e, list) else e for e in anchor_data.get("edges", []))
        positions = dict(anchor_data.get("positions", {}))
        metrics = dict(anchor_data.get("metrics", {}))

        # Apply each diff in order
        for diff_id in chain.diff_ids:
            parts = diff_id.rsplit("_block_", 1)
            if len(parts) != 2:
                continue

            diff_block = int(parts[1])
            diff = self.load_diff(base_sql_file, diff_block)

            if not diff:
                continue

            # Apply node changes
            for node_id in diff.added_nodes:
                nodes.add(node_id)
            for node_id in diff.removed_nodes:
                nodes.discard(node_id)
                positions.pop(node_id, None)
                metrics.pop(node_id, None)

            # Apply edge changes
            for edge in diff.added_edges:
                edges.add(tuple(edge))
            for edge in diff.removed_edges:
                edges.discard(tuple(edge))

            # Apply new positions
            positions.update(diff.new_positions)

            # Apply new metrics
            metrics.update(diff.new_metrics)

        reconstruction_time = (time.time() - start_time) * 1000

        return ReconstructedSnapshot(
            snapshot_id=snapshot_id,
            nodes=list(nodes),
            edges=list(edges),
            positions=positions,
            metrics=metrics,
            anchor_id=chain.anchor_id,
            chain_length=chain.chain_length,
            reconstruction_time_ms=reconstruction_time,
        )

    # =========================================================================
    # Index Management
    # =========================================================================

    def _load_diff_index(self, base_sql_file: str) -> Dict[str, Dict]:
        """Load the diff index for a base SQL file."""
        if base_sql_file in self._diff_index:
            return self._diff_index[base_sql_file]

        index_path = self.cache_dir / base_sql_file / "_diff_index.json"
        if not index_path.exists():
            self._diff_index[base_sql_file] = {}
            return {}

        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                self._diff_index[base_sql_file] = json.load(f)
        except (json.JSONDecodeError, IOError):
            self._diff_index[base_sql_file] = {}

        return self._diff_index[base_sql_file]

    def _save_diff_index(self, base_sql_file: str) -> None:
        """Save the diff index for a base SQL file."""
        if base_sql_file not in self._diff_index:
            return

        index_dir = self.cache_dir / base_sql_file
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / "_diff_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self._diff_index[base_sql_file], f, indent=2, default=str)

    def _update_diff_index(self, base_sql_file: str, metadata: DiffMetadata) -> None:
        """Update the diff index with new metadata."""
        if base_sql_file not in self._diff_index:
            self._load_diff_index(base_sql_file)

        self._diff_index[base_sql_file][metadata.snapshot_id] = {
            "snapshot_id": metadata.snapshot_id,
            "base_snapshot_id": metadata.base_snapshot_id,
            "is_anchor": metadata.is_anchor,
            "block_number": metadata.block_number,
            "added_node_count": metadata.added_node_count,
            "removed_node_count": metadata.removed_node_count,
            "added_edge_count": metadata.added_edge_count,
            "removed_edge_count": metadata.removed_edge_count,
            "created_at": metadata.created_at.isoformat() if metadata.created_at else None,
        }

        self._save_diff_index(base_sql_file)

    def _get_snapshot_dir(self, base_sql_file: str, block_number: int) -> Path:
        """Get the directory path for a snapshot."""
        return self.cache_dir / base_sql_file / f"block_{block_number}"

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_diff_storage_enabled(self) -> bool:
        """Check if diff storage is enabled in settings."""
        return getattr(settings, 'SNAPSHOT_DIFF_ENABLED', False)

    def get_storage_stats(self, base_sql_file: str) -> Dict[str, Any]:
        """
        Get storage statistics for a base SQL file.

        Returns stats about anchors, diffs, and storage savings.
        """
        diff_index = self._load_diff_index(base_sql_file)

        anchor_count = sum(1 for e in diff_index.values() if e.get("is_anchor", False))
        diff_count = len(diff_index) - anchor_count

        total_added_nodes = sum(e.get("added_node_count", 0) for e in diff_index.values())
        total_removed_nodes = sum(e.get("removed_node_count", 0) for e in diff_index.values())

        return {
            "base_sql_file": base_sql_file,
            "total_snapshots": len(diff_index),
            "anchor_count": anchor_count,
            "diff_count": diff_count,
            "total_added_nodes": total_added_nodes,
            "total_removed_nodes": total_removed_nodes,
        }
