"""
Arrow IPC Service

Converts graph data (nodes, edges, metrics, positions) into Apache Arrow
IPC streams for efficient binary transfer to the frontend.

Arrow IPC is ~10× smaller than JSON for numeric data and can be deserialized
with zero-copy on the frontend using Apache Arrow JS.
"""

import io
import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

logger = logging.getLogger(__name__)


class ArrowService:
    """Converts graph data to Arrow IPC binary streams."""

    def graph_elements_to_arrow(
        self,
        nodes: List[Dict[str, Any]],
        positions: Dict[str, Dict[str, float]],
        include_edges: bool = False,
        edges: Optional[List[tuple]] = None,
    ) -> bytes:
        """
        Convert graph nodes (with positions and metrics) to Arrow IPC bytes.

        Args:
            nodes: List of node dicts [{id, metric1, metric2, ...}, ...]
            positions: {node_id: {x, y}}
            include_edges: Whether to include edge data
            edges: List of (source, target) tuples

        Returns:
            Arrow IPC stream bytes containing a nodes table
        """
        # Build node records with positions
        records = []
        for node_data in nodes:
            node_id = str(node_data.get("id", ""))
            pos = positions.get(node_id, {})
            record = {
                "id": node_id,
                "x": float(pos.get("x", 0.0)),
                "y": float(pos.get("y", 0.0)),
            }
            # Add all other attributes (metrics, properties)
            for k, v in node_data.items():
                if k == "id":
                    continue
                if isinstance(v, (np.integer,)):
                    record[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    record[k] = float(v) if not np.isnan(v) else 0.0
                elif isinstance(v, (int, float, str, bool)):
                    record[k] = v
                elif v is None:
                    record[k] = 0.0
                # Skip complex types (lists, dicts)
            records.append(record)

        if not records:
            # Return empty table
            schema = pa.schema([
                ("id", pa.string()),
                ("x", pa.float32()),
                ("y", pa.float32()),
            ])
            table = pa.table({"id": [], "x": [], "y": []}, schema=schema)
        else:
            try:
                df = pd.DataFrame(records)
                # Coerce mixed-type columns to avoid Arrow serialization errors.
                # Columns with inconsistent types (e.g. str + NaN) fail in
                # pa.Table.from_pandas, so cast object columns with mixed
                # numeric/string values to string.
                for col in df.columns:
                    if col in ("id", "x", "y"):
                        continue
                    if df[col].dtype == object:
                        # Check if the column has mixed types
                        non_null = df[col].dropna()
                        if len(non_null) > 0:
                            types = set(type(v).__name__ for v in non_null)
                            if len(types) > 1:
                                df[col] = df[col].astype(str)
                            elif types == {"str"}:
                                df[col] = df[col].fillna("")
                            else:
                                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                        else:
                            df[col] = df[col].fillna(0.0)
                table = pa.Table.from_pandas(df, preserve_index=False)
            except Exception:
                # Fallback: return only id + positions (guaranteed safe)
                logger.warning(
                    "[ARROW] Full DataFrame conversion failed, falling back to id/x/y only",
                    exc_info=True,
                )
                safe_records = [{"id": r["id"], "x": r["x"], "y": r["y"]} for r in records]
                df = pd.DataFrame(safe_records)
                table = pa.Table.from_pandas(df, preserve_index=False)

        return self._table_to_ipc(table)

    def edges_to_arrow(
        self,
        edges: List[tuple],
        node_index: Optional[Dict[str, int]] = None,
        offset: int = 0,
        limit: int = 50000,
    ) -> bytes:
        """
        Convert graph edges to Arrow IPC bytes.

        Args:
            edges: List of (source, target) tuples
            node_index: Optional {node_id: integer_index} for pre-computed
                        cosmos.gl link indices
            offset: Start offset for pagination
            limit: Max edges to include

        Returns:
            Arrow IPC stream bytes containing an edges table
        """
        chunk = edges[offset:offset + limit]

        sources = [str(e[0]) for e in chunk]
        targets = [str(e[1]) for e in chunk]

        data = {
            "source": sources,
            "target": targets,
        }

        # Pre-compute integer indices for cosmos.gl
        if node_index is not None:
            data["source_idx"] = pa.array(
                [node_index.get(s, -1) for s in sources],
                type=pa.int32(),
            )
            data["target_idx"] = pa.array(
                [node_index.get(t, -1) for t in targets],
                type=pa.int32(),
            )

        table = pa.table(data)
        return self._table_to_ipc(table)

    def metrics_to_arrow(self, metrics_df: pd.DataFrame) -> bytes:
        """
        Convert a metrics DataFrame to Arrow IPC bytes.

        Args:
            metrics_df: DataFrame with node IDs as index/column and metric columns

        Returns:
            Arrow IPC stream bytes
        """
        if metrics_df is None or metrics_df.empty:
            schema = pa.schema([("id", pa.string())])
            table = pa.table({"id": []}, schema=schema)
            return self._table_to_ipc(table)

        # Ensure ID column exists
        df = metrics_df.copy()
        if "avatar" in df.columns:
            df = df.rename(columns={"avatar": "id"})
        elif df.index.name and df.index.name != "id":
            df = df.reset_index()
            if df.columns[0] != "id":
                df = df.rename(columns={df.columns[0]: "id"})

        # Replace inf/nan
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)

        table = pa.Table.from_pandas(df, preserve_index=False)
        return self._table_to_ipc(table)

    @staticmethod
    def _table_to_ipc(table: pa.Table) -> bytes:
        """Serialize Arrow Table to IPC stream bytes."""
        sink = io.BytesIO()
        with ipc.new_stream(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
        return sink.getvalue()


# Singleton
arrow_service = ArrowService()
