import tempfile
import unittest
from pathlib import Path

from backend.services.duckdb_service import DuckDBService


class DuckDBAtomicWriteTests(unittest.TestCase):
    def test_write_positions_atomic_replaces_final_file(self):
        service = DuckDBService()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layout.parquet"

            service.write_positions_atomic(
                {"old": {"x": 1.0, "y": 2.0}},
                path,
            )
            service.write_positions_atomic(
                {"new": {"x": 3.0, "y": 4.0}},
                path,
            )

            self.assertEqual(
                service.read_positions(path),
                {"new": {"x": 3.0, "y": 4.0}},
            )


if __name__ == "__main__":
    unittest.main()
