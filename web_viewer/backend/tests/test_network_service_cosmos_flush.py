import importlib
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.config import settings
from backend.models.requests import LoadConfig
from backend.services.network_service import NetworkService
import pandas as pd

network_service_module = importlib.import_module("backend.services.network_service")


class CosmosFlushTests(unittest.TestCase):
    def test_flush_reads_layout_under_lock_and_writes_snapshot_copy(self):
        service = NetworkService()
        graph_id = "graph"

        service.layouts[graph_id] = {
            "n1": {"x": 1.0, "y": 2.0},
        }
        service._cosmos_dirty_count[graph_id] = settings.COSMOS_PERSIST_MIN_DIRTY

        def save_side_effect(graph_id_arg, positions):
            self.assertEqual(graph_id_arg, graph_id)
            self.assertIsNot(positions, service.layouts[graph_id])
            self.assertEqual(positions["n1"]["x"], 1.0)

            service.layouts[graph_id]["n1"]["x"] = 999.0
            self.assertEqual(positions["n1"]["x"], 1.0)

        service.cache_service.save_cosmos_live_layout = MagicMock(side_effect=save_side_effect)

        service._flush_cosmos_positions(graph_id)

        service.cache_service.save_cosmos_live_layout.assert_called_once()
        self.assertEqual(service._cosmos_dirty_count[graph_id], 0)

    def test_flush_all_cancels_timers_before_writing(self):
        service = NetworkService()
        graph_id = "graph"
        events = []

        service.layouts[graph_id] = {
            "n1": {"x": 1.0, "y": 2.0},
        }
        service._cosmos_dirty_count[graph_id] = settings.COSMOS_PERSIST_MIN_DIRTY

        timer = MagicMock()
        timer.cancel.side_effect = lambda: events.append("cancel")
        service._cosmos_flush_timer[graph_id] = timer

        def save_side_effect(graph_id_arg, positions):
            self.assertEqual(graph_id_arg, graph_id)
            self.assertEqual(positions["n1"]["x"], 1.0)
            events.append("write")

        service.cache_service.save_cosmos_live_layout = MagicMock(side_effect=save_side_effect)

        service.flush_all_cosmos_positions()

        self.assertEqual(events, ["cancel", "write"])
        self.assertNotIn(graph_id, service._cosmos_flush_timer)
        self.assertEqual(service._cosmos_dirty_count[graph_id], 0)

    def test_load_network_defers_layout_to_frontend_when_backend_layout_disabled(self):
        service = NetworkService()
        original_setting = settings.BACKEND_LAYOUT_ON_LOAD
        settings.BACKEND_LAYOUT_ON_LOAD = False

        try:
            service.load_edge_layers_from_sql = MagicMock(return_value={
                "graph": pd.DataFrame([
                    {"source": "a", "target": "b"},
                    {"source": "b", "target": "c"},
                ])
            })
            service.load_api_properties = MagicMock(return_value=(pd.DataFrame(), {}, None))
            service.layout_service.compute_layout = MagicMock()
            service.cache_service.get_resume_layout = MagicMock(return_value=None)

            result = service.load_network(LoadConfig(
                sql_files=["graph.sql"],
                use_cached_layout=True,
                skip_sql=False,
                preset=None,
                categories=None,
                metrics=None,
            ))

            service.layout_service.compute_layout.assert_not_called()
            self.assertEqual(result.layout_algorithm, "frontend_deferred")
            self.assertEqual(service.layouts["graph"], {})
        finally:
            settings.BACKEND_LAYOUT_ON_LOAD = original_setting

    def test_load_node_properties_falls_back_to_native_postgres_for_postgres_only_sql(self):
        service = NetworkService()
        sql_path = settings.NODE_PROPERTIES_DIR / "crc_v1_avatars.sql"
        original_exists = Path.exists

        duckdb_mock = MagicMock(side_effect=Exception(
            "Catalog Error: Scalar Function with name array_remove does not exist!"
        ))
        native_df = pd.DataFrame([{"avatar": "Alice", "flag": True}])
        native_mock = MagicMock(return_value=native_df)
        original_session = network_service_module._db.session
        original_native = network_service_module._db.execute_postgres_sql_native

        fake_session = MagicMock()
        fake_session.attach_postgres = MagicMock()
        def fake_execute(sql):
            if str(sql).startswith("SET threads ="):
                return MagicMock()
            return duckdb_mock(sql)
        fake_session.execute = MagicMock(side_effect=fake_execute)

        fake_session_cm = MagicMock()
        fake_session_cm.__enter__.return_value = fake_session
        fake_session_cm.__exit__.return_value = None

        def fake_exists(path_obj):
            if path_obj == sql_path:
                return True
            return original_exists(path_obj)

        network_service_module._db.session = MagicMock(return_value=fake_session_cm)
        network_service_module._db.execute_postgres_sql_native = native_mock

        try:
            with unittest.mock.patch.object(Path, "exists", fake_exists):
                result = service.load_node_properties_from_sql(["crc_v1_avatars.sql"], skip_sql=False)
        finally:
            network_service_module._db.session = original_session
            network_service_module._db.execute_postgres_sql_native = original_native

        fake_session.attach_postgres.assert_called_once()
        duckdb_mock.assert_called_once()
        native_mock.assert_called_once()
        self.assertIn("v1", result)
        self.assertEqual(result["v1"]["avatar"].tolist(), ["alice"])

    def test_load_edge_layers_retries_on_too_many_connections(self):
        service = NetworkService()
        sql_path = settings.SQL_DIR / "graph.sql"
        original_exists = Path.exists
        original_session = network_service_module._db.session
        original_sleep = network_service_module.time.sleep
        original_attempts = settings.POSTGRES_RETRY_ATTEMPTS
        original_delay = settings.POSTGRES_RETRY_BASE_DELAY_S

        settings.POSTGRES_RETRY_ATTEMPTS = 2
        settings.POSTGRES_RETRY_BASE_DELAY_S = 0.01

        first_session = MagicMock()
        first_session.attach_postgres = MagicMock()
        def first_execute(sql):
            if str(sql).startswith("SET threads ="):
                return MagicMock()
            raise Exception('FATAL: too many connections for role "analytics_trust"')
        first_session.execute = MagicMock(side_effect=first_execute)
        first_cm = MagicMock()
        first_cm.__enter__.return_value = first_session
        first_cm.__exit__.return_value = None

        second_relation = MagicMock()
        second_relation.fetchdf.return_value = pd.DataFrame([{"source": "a", "target": "b"}])
        second_session = MagicMock()
        second_session.attach_postgres = MagicMock()
        def second_execute(sql):
            if str(sql).startswith("SET threads ="):
                return MagicMock()
            return second_relation
        second_session.execute = MagicMock(side_effect=second_execute)
        second_cm = MagicMock()
        second_cm.__enter__.return_value = second_session
        second_cm.__exit__.return_value = None

        session_factory = MagicMock(side_effect=[first_cm, second_cm])
        sleep_mock = MagicMock()

        def fake_exists(path_obj):
            if path_obj == sql_path:
                return True
            return original_exists(path_obj)

        network_service_module._db.session = session_factory
        network_service_module.time.sleep = sleep_mock

        try:
            with unittest.mock.patch.object(Path, "exists", fake_exists):
                with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="SELECT 1")):
                    result = service.load_edge_layers_from_sql(["graph.sql"])
        finally:
            network_service_module._db.session = original_session
            network_service_module.time.sleep = original_sleep
            settings.POSTGRES_RETRY_ATTEMPTS = original_attempts
            settings.POSTGRES_RETRY_BASE_DELAY_S = original_delay

        self.assertIn("graph", result)
        self.assertEqual(len(result["graph"]), 1)
        self.assertEqual(session_factory.call_count, 2)
        first_session.attach_postgres.assert_called_once()
        second_session.attach_postgres.assert_called_once()
        sleep_mock.assert_called_once_with(0.01)
        first_cm.__exit__.assert_called()
        second_cm.__exit__.assert_called()


if __name__ == "__main__":
    unittest.main()
