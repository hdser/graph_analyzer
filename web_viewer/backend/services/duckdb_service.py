"""
DuckDB Engine Service

Central data engine replacing Pandas for all Parquet I/O, SQL database reads,
joins, and user-facing SQL queries. DuckDB reads Parquet natively via Arrow,
is multi-threaded, and handles concurrent requests via stateless :memory: connections.

Concurrency model:
- Each operation gets an isolated :memory: DuckDB instance
- No persistent .duckdb file, no file locks
- Multiple workers/users safely read Parquet files concurrently (OS-level file reads)
- Connection created and destroyed per-operation in try/finally

Two connection modes:
- Internal: generous limits for data pipeline operations
- User-facing: sandboxed with restricted memory/CPU/access
"""

import io
from pathlib import Path
from typing import Optional, List, Dict, Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

from ..config import settings


class DuckDBService:
    """DuckDB-based data engine for Parquet I/O and SQL queries."""

    def __init__(self):
        self.data_dir = settings.DATA_CACHE_DIR
        self.layouts_dir = settings.LAYOUTS_DIR

    # =========================================================================
    # Internal Connection (for data pipeline)
    # =========================================================================

    def _create_internal_connection(self) -> duckdb.DuckDBPyConnection:
        """Create an internal DuckDB connection with generous limits."""
        conn = duckdb.connect(database=':memory:')
        conn.execute(f"SET memory_limit = '{settings.DUCKDB_MEMORY_LIMIT}'")
        return conn

    # =========================================================================
    # Parquet I/O
    # =========================================================================

    def read_parquet(self, path: Path) -> pd.DataFrame:
        """
        Read a Parquet file into a Pandas DataFrame via DuckDB.

        Args:
            path: Path to the Parquet file

        Returns:
            Pandas DataFrame
        """
        conn = self._create_internal_connection()
        try:
            return conn.execute(
                f"SELECT * FROM read_parquet('{path}')"
            ).fetchdf()
        finally:
            conn.close()

    def read_parquet_arrow(self, path: Path) -> pa.Table:
        """
        Read a Parquet file into an Arrow Table (zero-copy).

        Args:
            path: Path to the Parquet file

        Returns:
            PyArrow Table
        """
        conn = self._create_internal_connection()
        try:
            result = conn.execute(
                f"SELECT * FROM read_parquet('{path}')"
            ).arrow()
            if hasattr(result, 'read_all'):
                result = result.read_all()
            return result
        finally:
            conn.close()

    def read_parquet_columns(
        self,
        path: Path,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Read specific columns from a Parquet file.

        Args:
            path: Path to the Parquet file
            columns: List of column names to read

        Returns:
            Pandas DataFrame with selected columns
        """
        cols = ", ".join(f'"{c}"' for c in columns)
        conn = self._create_internal_connection()
        try:
            return conn.execute(
                f"SELECT {cols} FROM read_parquet('{path}')"
            ).fetchdf()
        finally:
            conn.close()

    def read_parquet_row_count(self, path: Path) -> int:
        """
        Get row count from a Parquet file without loading all data.

        Args:
            path: Path to the Parquet file

        Returns:
            Number of rows
        """
        conn = self._create_internal_connection()
        try:
            result = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{path}')"
            ).fetchone()
            return result[0]
        finally:
            conn.close()

    def write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """
        Write a DataFrame to Parquet via DuckDB.

        Args:
            df: Pandas DataFrame to write
            path: Output Parquet file path
        """
        conn = self._create_internal_connection()
        try:
            conn.execute(
                f"COPY df TO '{path}' (FORMAT PARQUET, COMPRESSION SNAPPY)"
            )
        finally:
            conn.close()

    # =========================================================================
    # Layout-specific I/O (optimized)
    # =========================================================================

    def read_positions(self, path: Path) -> Dict[str, Dict[str, float]]:
        """
        Read layout positions from Parquet directly as a dict.

        Optimized: extracts columns as lists instead of iterating rows.

        Args:
            path: Path to layout Parquet file (columns: node_id, x, y)

        Returns:
            Dict of {node_id: {x, y}}
        """
        conn = self._create_internal_connection()
        try:
            result = conn.execute(
                f"SELECT CAST(node_id AS VARCHAR) AS node_id, "
                f"CAST(x AS DOUBLE) AS x, CAST(y AS DOUBLE) AS y "
                f"FROM read_parquet('{path}')"
            ).fetchall()
            return {
                row[0]: {'x': row[1], 'y': row[2]}
                for row in result
            }
        finally:
            conn.close()

    def write_positions(
        self,
        positions: Dict[str, Dict[str, float]],
        path: Path
    ) -> None:
        """
        Write layout positions dict to Parquet.

        Args:
            positions: Dict of {node_id: {x, y}}
            path: Output Parquet file path
        """
        rows = [
            {'node_id': node_id, 'x': pos['x'], 'y': pos['y']}
            for node_id, pos in positions.items()
        ]
        df = pd.DataFrame(rows)
        self.write_parquet(df, path)

    # =========================================================================
    # SQL Database Reads (via postgres_scanner)
    # =========================================================================

    def execute_postgres_sql(
        self,
        sql_query: str,
        connection_uri: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute SQL against PostgreSQL via DuckDB's postgres_scanner.

        Args:
            sql_query: SQL query to execute
            connection_uri: PostgreSQL connection URI (defaults to settings.database_url)

        Returns:
            Pandas DataFrame with query results
        """
        uri = connection_uri or settings.database_url
        conn = self._create_internal_connection()
        try:
            conn.install_extension("postgres_scanner")
            conn.load_extension("postgres_scanner")
            conn.execute(f"ATTACH '{uri}' AS pg (TYPE POSTGRES, READ_ONLY)")
            # Set default catalog to attached postgres so unqualified table
            # references resolve correctly (SQL templates written for postgres)
            conn.execute("USE pg")
            result = conn.execute(sql_query).fetchdf()
            return result
        finally:
            conn.close()

    # =========================================================================
    # Parquet Joins (faster than pd.merge)
    # =========================================================================

    def join_parquet_files(
        self,
        left_path: Path,
        right_path: Path,
        on: str,
        how: str = "left"
    ) -> pd.DataFrame:
        """
        Join two Parquet files via SQL.

        Args:
            left_path: Path to left Parquet file
            right_path: Path to right Parquet file
            on: Column name to join on
            how: Join type ('left', 'right', 'inner', 'full')

        Returns:
            Pandas DataFrame with join result
        """
        join_type = {
            'left': 'LEFT JOIN',
            'right': 'RIGHT JOIN',
            'inner': 'INNER JOIN',
            'full': 'FULL OUTER JOIN',
            'outer': 'FULL OUTER JOIN',
        }.get(how, 'LEFT JOIN')

        conn = self._create_internal_connection()
        try:
            return conn.execute(f"""
                SELECT l.*, r.*
                FROM read_parquet('{left_path}') l
                {join_type} read_parquet('{right_path}') r
                ON l."{on}" = r."{on}"
            """).fetchdf()
        finally:
            conn.close()

    # =========================================================================
    # User-facing SQL (sandboxed)
    # =========================================================================

    def _create_user_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Create a sandboxed DuckDB connection for user queries.

        Security:
        - enable_external_access = false (no reading server files)
        - Memory limited to prevent OOM from expensive JOINs
        - Thread limited to prevent CPU monopolization
        """
        conn = duckdb.connect(database=':memory:')
        conn.execute(f"SET memory_limit = '{settings.DUCKDB_USER_MEMORY_LIMIT}'")
        conn.execute(f"SET threads = {settings.DUCKDB_USER_THREADS}")

        # Load all Parquet files into in-memory tables.
        # We must materialize the data (CREATE TABLE ... AS SELECT) rather
        # than using views, because views are lazy and need file access at
        # query time — which would be blocked after we disable external access.
        if self.data_dir.exists():
            for parquet_file in self.data_dir.glob("*.parquet"):
                table_name = parquet_file.stem
                try:
                    conn.execute(
                        f'CREATE TABLE "{table_name}" AS '
                        f"SELECT * FROM read_parquet('{parquet_file}')"
                    )
                except Exception:
                    pass

        if self.layouts_dir.exists():
            for parquet_file in self.layouts_dir.glob("*.parquet"):
                table_name = f"layout_{parquet_file.stem}"
                try:
                    conn.execute(
                        f'CREATE TABLE "{table_name}" AS '
                        f"SELECT * FROM read_parquet('{parquet_file}')"
                    )
                except Exception:
                    pass

        # Sandbox: disable external access AFTER data is loaded
        # User queries can only read the pre-loaded in-memory tables
        conn.execute("SET enable_external_access = false")

        return conn

    def _validate_user_sql(self, sql: str) -> None:
        """Validate that user SQL is a SELECT query only."""
        sql_upper = sql.strip().upper()
        forbidden = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER',
            'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE', 'ATTACH',
            'COPY', 'EXPORT', 'IMPORT', 'LOAD', 'INSTALL'
        ]
        for keyword in forbidden:
            if sql_upper.startswith(keyword):
                raise ValueError(
                    f"Only SELECT queries are allowed. "
                    f"'{keyword}' statements are forbidden."
                )

    def execute_user_query(
        self,
        sql: str,
        max_rows: Optional[int] = None
    ) -> pa.Table:
        """
        Execute a user SQL query and return Arrow table.

        Args:
            sql: SQL query (SELECT only)
            max_rows: Maximum rows to return

        Returns:
            PyArrow Table

        Raises:
            ValueError: If query is not SELECT
        """
        self._validate_user_sql(sql)
        max_rows = max_rows or settings.DUCKDB_MAX_ROWS

        # Add LIMIT if not present
        sql_upper = sql.strip().upper()
        if 'LIMIT' not in sql_upper:
            sql = f"SELECT * FROM ({sql}) _sub LIMIT {max_rows}"

        conn = self._create_user_connection()
        try:
            result = conn.execute(sql).arrow()
            # DuckDB may return RecordBatchReader instead of Table
            if hasattr(result, 'read_all'):
                result = result.read_all()
            return result
        finally:
            conn.close()

    def execute_user_query_json(
        self,
        sql: str,
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute user query and return JSON-serializable result.

        Args:
            sql: SQL query
            max_rows: Maximum rows

        Returns:
            Dict with columns, rows, row_count, schema
        """
        table = self.execute_user_query(sql, max_rows)
        df = table.to_pandas()

        # Clean numpy types for JSON serialization
        rows = []
        for _, row in df.iterrows():
            clean_row = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val) if not np.isnan(val) else None
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                elif pd.isna(val):
                    val = None
                clean_row[col] = val
            rows.append(clean_row)

        return {
            "columns": list(df.columns),
            "rows": rows,
            "row_count": len(rows),
            "schema": [
                {"name": f.name, "type": str(f.type)}
                for f in table.schema
            ]
        }

    def execute_user_query_arrow(
        self,
        sql: str,
        max_rows: Optional[int] = None
    ) -> bytes:
        """
        Execute user query and return Arrow IPC bytes.

        Args:
            sql: SQL query
            max_rows: Maximum rows

        Returns:
            Arrow IPC stream bytes
        """
        table = self.execute_user_query(sql, max_rows)
        sink = io.BytesIO()
        with ipc.new_stream(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        return sink.getvalue()

    def list_tables(self) -> List[Dict[str, Any]]:
        """
        List all available virtual tables from Parquet cache.

        Returns:
            List of table metadata with name, columns, and row counts
        """
        conn = self._create_user_connection()
        try:
            tables = []
            result = conn.execute("SHOW TABLES").fetchall()
            for (name,) in result:
                try:
                    schema = conn.execute(f'DESCRIBE "{name}"').fetchall()
                    count = conn.execute(
                        f'SELECT COUNT(*) FROM "{name}"'
                    ).fetchone()[0]
                    tables.append({
                        "name": name,
                        "columns": [
                            {"name": col, "type": dtype}
                            for col, dtype, *_ in schema
                        ],
                        "row_count": count,
                    })
                except Exception:
                    pass  # Skip tables that can't be described
            return tables
        finally:
            conn.close()
