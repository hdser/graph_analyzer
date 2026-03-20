import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backend.models.snapshot import LayoutSource, MetricsMode, SnapshotMetadata, SnapshotStatus
from backend.services.snapshot_service import SnapshotService
from backend.services.snapshot_storage import SnapshotStorage


class SnapshotCompareTests(unittest.TestCase):
    def _metadata(self, base_sql_file: str, block_number: int, node_count: int, edge_count: int) -> SnapshotMetadata:
        return SnapshotMetadata(
            snapshot_id=f"{base_sql_file}_block_{block_number}",
            base_sql_file=base_sql_file,
            block_number=block_number,
            block_timestamp=datetime(2024, 1, block_number),
            label=f"block {block_number}",
            node_count=node_count,
            edge_count=edge_count,
            metrics_computed=[],
            metrics_mode=MetricsMode.NONE.value,
            layout_source=LayoutSource.MASTER.value,
            layout_unknown_nodes=0,
            created_at=datetime.utcnow(),
            computation_time_seconds=0.1,
            checksums={},
            status=SnapshotStatus.READY.value,
        )

    def test_compare_snapshots_reconstructed_edges_are_materialized_in_same_session(self):
        base_sql_file = "graph"

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SnapshotStorage(cache_dir=Path(tmpdir))
            service = SnapshotService(storage=storage)

            from_edges = pd.DataFrame([{"source": "a", "target": "b"}])
            from_layout = {
                "a": {"x": 0.0, "y": 0.0},
                "b": {"x": 1.0, "y": 1.0},
            }
            storage.save_snapshot(
                base_sql_file=base_sql_file,
                block_number=1,
                edges_df=from_edges,
                layout=from_layout,
                metrics_df=None,
                metadata=self._metadata(base_sql_file, 1, 2, 1),
            )

            to_edges = pd.DataFrame([
                {"source": "a", "target": "b"},
                {"source": "b", "target": "c"},
            ])
            to_layout = {
                "a": {"x": 0.0, "y": 0.0},
                "b": {"x": 1.0, "y": 1.0},
                "c": {"x": 2.0, "y": 2.0},
            }
            storage.save_snapshot(
                base_sql_file=base_sql_file,
                block_number=2,
                edges_df=to_edges,
                layout=to_layout,
                metrics_df=None,
                metadata=self._metadata(base_sql_file, 2, 3, 2),
            )

            edges_path = storage.get_snapshot_dir(base_sql_file, 2) / "edges.parquet"
            edges_path.unlink()

            reconstructed = SimpleNamespace(
                edges=[
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "c"},
                ]
            )

            with patch.object(storage, "load_snapshot_with_diff", return_value=reconstructed) as load_mock:
                result = service.compare_snapshots(base_sql_file, 1, 2)

            load_mock.assert_called_once_with(base_sql_file, 2)
            self.assertEqual(result["diff"]["added_edge_count"], 1)
            self.assertEqual(result["diff"]["removed_edge_count"], 0)
            self.assertEqual(result["diff"]["added_node_count"], 1)
            self.assertEqual(result["layout"]["c"], {"x": 2.0, "y": 2.0})


if __name__ == "__main__":
    unittest.main()
