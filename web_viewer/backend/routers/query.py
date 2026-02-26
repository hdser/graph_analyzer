"""
Query Router

API endpoints for user SQL queries against cached Parquet data via DuckDB.
All queries run in sandboxed :memory: connections with resource limits.
"""

import time
from typing import Optional, Dict, List, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from ..services.duckdb_service import DuckDBService

_db = DuckDBService()

router = APIRouter(prefix="/api/query", tags=["query"])


class QueryRequest(BaseModel):
    sql: str
    max_rows: Optional[int] = None


@router.post("/sql")
def execute_sql(request: QueryRequest) -> Dict[str, Any]:
    """
    Execute a user SQL query and return JSON results.

    Only SELECT queries are allowed. Queries run in a sandboxed DuckDB
    connection with restricted memory, threads, and no external file access.
    """
    start = time.time()

    try:
        result = _db.execute_user_query_json(request.sql, request.max_rows)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["execution_time_ms"] = round((time.time() - start) * 1000, 1)
    return result


@router.post("/arrow")
def execute_sql_arrow(request: QueryRequest) -> Response:
    """
    Execute a user SQL query and return Arrow IPC bytes.

    Same sandboxing as /sql, but returns binary Arrow IPC stream
    for efficient transfer of large result sets.
    """
    try:
        arrow_bytes = _db.execute_user_query_arrow(request.sql, request.max_rows)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=arrow_bytes,
        media_type="application/vnd.apache.arrow.stream",
    )


@router.get("/tables")
def list_tables() -> List[Dict[str, Any]]:
    """
    List all available virtual tables from cached Parquet files.

    Returns table name, columns (with types), and row counts.
    """
    try:
        return _db.list_tables()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{table_name}")
def get_table_schema(table_name: str) -> Dict[str, Any]:
    """
    Get detailed schema for a specific table.
    """
    try:
        tables = _db.list_tables()
        for table in tables:
            if table["name"] == table_name:
                return table
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
