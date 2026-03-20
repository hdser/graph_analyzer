import importlib
import unittest
from unittest.mock import patch

import pandas as pd

from backend.models.snapshot import MetricsMode
from backend.services.snapshot_service import SnapshotService

snapshot_service_module = importlib.import_module("backend.services.snapshot_service")


class SnapshotMetricsGraphBuildTests(unittest.TestCase):
    def test_from_pandas_edgelist_path_can_add_explicit_isolated_nodes_when_node_set_present(self):
        service = SnapshotService()
        edges_df = pd.DataFrame([
            {"source": "a", "target": "b"},
        ])
        captured_graph = {}

        class FakeMetricEngine:
            def __init__(self, graph):
                captured_graph["graph"] = graph

            def compute(self, preset=None, categories=None):
                return pd.DataFrame([
                    {"avatar": "a", "in_degree": 0},
                    {"avatar": "b", "in_degree": 1},
                    {"avatar": "isolated", "in_degree": 0},
                ])

        with patch.object(snapshot_service_module, "MetricEngine", FakeMetricEngine):
            metrics_df, metric_names = service._compute_snapshot_metrics(
                edges_df,
                MetricsMode.BASIC,
                snapshot_nodes={"a", "b", "isolated"},
            )

        graph = captured_graph["graph"]
        self.assertTrue(graph.has_node("isolated"))
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertIsNotNone(metrics_df)
        self.assertIn("in_degree", metric_names)


if __name__ == "__main__":
    unittest.main()
