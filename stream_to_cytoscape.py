"""
Streaming / multi-network driver for Cytoscape.

Features:
- Multiple SQL files, each defining ONE network (edge set + optional edge attributes).
- All networks share the same set of avatars (node IDs).
- Metrics are computed ONCE on a chosen base graph (metrics_graph_id),
  then applied to ALL avatars in ALL networks.
- Two run modes:
    --run-type once   -> build all networks once
    --run-type stream -> periodically recompute edges + metrics and
                         update/recreate networks as needed.
"""

import os
import time
import argparse
import pandas as pd
import networkx as nx

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

import py4cytoscape as p4c
from py4cytoscape import CyError

from graph_metrics import GraphMetrics  # uses your big metrics engine


# ======================================================================
# 1. Load multiple edge layers from SQL files
# ======================================================================

def load_edge_layers_from_sql(sql_files: str) -> dict:
    """
    Load multiple edge layers from a comma-separated list of .sql files.

    Each SQL file:
      - Is executed against the configured PostgreSQL DB
      - Must return at least: source, target
      - May return extra edge attributes (flow, weight, etc.)

    Returns:
      dict[str, pd.DataFrame]:
        {
          graph_id_1: df_edges_1,
          graph_id_2: df_edges_2,
          ...
        }

      where graph_id = basename of the SQL file without extension, e.g.
        'sql/crc_v2_trust.sql' -> 'crc_v2_trust'
    """
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_name]):
        raise ValueError("Missing DB credentials: DB_USER, DB_PASSWORD, DB_HOST, DB_NAME must be set in .env")

    url = URL.create(
        "postgresql+psycopg2",
        username=db_user,
        password=db_password,
        host=db_host,
        database=db_name,
    )
    engine = create_engine(url)

    file_list = [p.strip() for p in sql_files.split(",") if p.strip()]
    if not file_list:
        raise ValueError("No SQL files provided to --sql-files")

    edge_layers: dict[str, pd.DataFrame] = {}

    for path in file_list:
        graph_id = os.path.splitext(os.path.basename(path))[0]
        print(f"[LOAD] Executing SQL from file: {path} (graph_id={graph_id})")

        with open(path, "r") as f:
            query = f.read()

        start = time.time()
        df = pd.read_sql_query(query, engine)
        elapsed = time.time() - start

        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError(f"SQL file {path} must return 'source' and 'target' columns")

        print(f"[LOAD]   -> {len(df):,} edges loaded in {elapsed:.2f}s")
        edge_layers[graph_id] = df

    engine.dispose()

    print(f"[LOAD] Loaded {len(edge_layers)} edge layers: {list(edge_layers.keys())}")
    return edge_layers


# ======================================================================
# 2. Compute metrics on ONE layer, then expand to ALL avatars
# ======================================================================

def compute_metrics_for_shared_avatars(edge_layers: dict,
                                       metrics_graph_id: str,
                                       metrics_mode: str | None = None,
                                       n_jobs: int | None = None) -> pd.DataFrame:
    """
    Compute node metrics on ONE chosen layer, then expand to ALL avatars.

    Args:
      edge_layers: dict[str, pd.DataFrame]
        Output of load_edge_layers_from_sql(), one df per graph_id.
      metrics_graph_id: str
        Which graph_id to use for metrics computation (e.g. 'crc_v2_trust').
      metrics_mode: str | None
        Passed to GraphMetrics (e.g. 'basic', 'topology,community', ...).
        If None, GraphMetrics defaults to 'all'.
      n_jobs: int | None
        Number of parallel workers; if None, GraphMetrics uses its default.

    Returns:
      df_metrics_all: pd.DataFrame
        One row per avatar across ALL layers, with:
          - 'avatar'
          - metric_1, metric_2, ...
    """
    import numpy as np

    if metrics_graph_id not in edge_layers:
        raise ValueError(f"metrics_graph_id='{metrics_graph_id}' not found in edge_layers keys: {list(edge_layers.keys())}")

    df_metrics_edges = edge_layers[metrics_graph_id]
    print(f"[METRICS] Using graph_id='{metrics_graph_id}' for metrics ({len(df_metrics_edges):,} edges)")

    G = nx.DiGraph()
    G.add_edges_from(df_metrics_edges[["source", "target"]].itertuples(index=False, name=None))

    # Compute metrics for nodes in this graph
    metrics_calc = GraphMetrics(G, n_jobs=n_jobs, metrics_mode=metrics_mode or "all")
    df_metrics = metrics_calc.compute_all()  # ['avatar', metric1, metric2, ...]

    # Build union of all avatars across ALL layers
    all_avatars = set(df_metrics["avatar"].unique())
    for gid, df_edges in edge_layers.items():
        all_avatars.update(df_edges["source"].unique())
        all_avatars.update(df_edges["target"].unique())

    print(f"[METRICS] Total unique avatars across all layers: {len(all_avatars):,}")

    # Create a full metrics table for all avatars
    df_all = pd.DataFrame({"avatar": list(all_avatars)})
    df_metrics_all = df_all.merge(df_metrics, on="avatar", how="left")

    # Fill NaNs / inf for avatars not in metrics base graph
    metric_cols = [c for c in df_metrics_all.columns if c != "avatar"]
    df_metrics_all[metric_cols] = df_metrics_all[metric_cols].replace([np.inf, -np.inf], 0).fillna(0)

    print(f"[METRICS] Metrics computed for {len(df_metrics_all):,} avatars (shared across all networks)")
    return df_metrics_all


# ======================================================================
# 3. Cytoscape network creation & updates
# ======================================================================

def create_cytoscape_network_for_layer(graph_id: str,
                                       df_edges: pd.DataFrame,
                                       df_metrics_all: pd.DataFrame,
                                       collection: str = "Trust Networks") -> int:
    """
    Create a Cytoscape network for a single graph layer.

    - Nodes: all avatars from df_metrics_all (shared node table).
    - Edges: only edges from this layer (df_edges).
    - Node attributes: metrics from df_metrics_all (same for all networks).

    Returns:
      net_suid: int
        SUID of the created Cytoscape network.
    """
    p4c.cytoscape_ping()

    # Nodes: shared avatar universe
    df_nodes = df_metrics_all.rename(columns={"avatar": "id"}).copy()

    # Edges: this specific layer
    df_edges_stream = df_edges.copy()

    # Optional interaction column: use graph_id if not already present
    if "interaction" not in df_edges_stream.columns:
        df_edges_stream["interaction"] = graph_id

    title = f"{graph_id} (shared metrics)"
    net_suid = p4c.create_network_from_data_frames(
        nodes=df_nodes,
        edges=df_edges_stream,
        title=title,
        collection=collection
    )

    # Ensure view exists and layout is applied
    try:
        views = p4c.get_network_views(net_suid)
        if not views:
            p4c.create_network_view(net_suid)
    except CyError as e:
        print(f"[CYTOSCAPE] Warning: could not ensure network view for {graph_id}: {e}")

    try:
        p4c.set_current_network(net_suid)
        p4c.layout_network("force-directed", network=net_suid)
    except CyError as e:
        print(f"[CYTOSCAPE] Warning: layout failed for {graph_id}: {e}")

    print(f"[CYTOSCAPE] Created network for graph_id='{graph_id}' with SUID={net_suid}")
    return net_suid


def update_node_metrics_only(net_suid: int, df_metrics_all: pd.DataFrame) -> None:
    """
    Update node attributes (metrics) in an existing Cytoscape network.

    - Assumes:
        * Cytoscape node key column is 'name'
        * 'name' matches 'avatar' values (since we created nodes with id=avatar)
    """
    try:
        p4c.set_current_network(net_suid)
        p4c.load_table_data(
            df_metrics_all,
            data_key_column="avatar",  # DataFrame column
            table_key_column="name",   # Cytoscape node key
            table="node"
        )
    except CyError as e:
        print(f"[CYTOSCAPE] Warning: failed to update node metrics for network {net_suid}: {e}")


# ======================================================================
# 4. Helper: compare edge sets (for streaming)
# ======================================================================

def same_edge_set(df_old: pd.DataFrame,
                  df_new: pd.DataFrame,
                  edge_cols=("source", "target")) -> bool:
    """
    Return True if the edge set is identical between two DataFrames.

    Only compares specified columns (default: 'source', 'target').
    Order does NOT matter.
    """
    cols = [c for c in edge_cols if c in df_old.columns and c in df_new.columns]
    if not cols:
        return False

    if len(df_old) != len(df_new):
        return False

    a = df_old[cols].copy().sort_values(cols).reset_index(drop=True)
    b = df_new[cols].copy().sort_values(cols).reset_index(drop=True)
    return a.equals(b)


# ======================================================================
# 5. High-level: Build all networks once
# ======================================================================

def build_networks_with_shared_metrics(sql_files: str,
                                       metrics_graph_id: str,
                                       metrics_mode: str | None = None,
                                       n_jobs: int | None = None,
                                       collection: str = "Trust Networks") -> dict:
    """
    One-shot pipeline:

      1. Load all edge layers from the given SQL files.
      2. Compute node metrics on ONE selected layer (metrics_graph_id).
      3. Create a separate Cytoscape network for EACH layer,
         all sharing the same node metrics.

    Returns:
      networks: dict[str, int]
        Mapping from graph_id -> Cytoscape network SUID
    """
    edge_layers = load_edge_layers_from_sql(sql_files)

    df_metrics_all = compute_metrics_for_shared_avatars(
        edge_layers=edge_layers,
        metrics_graph_id=metrics_graph_id,
        metrics_mode=metrics_mode,
        n_jobs=n_jobs
    )

    networks: dict[str, int] = {}
    for gid, df_edges in edge_layers.items():
        net_suid = create_cytoscape_network_for_layer(
            graph_id=gid,
            df_edges=df_edges,
            df_metrics_all=df_metrics_all,
            collection=collection
        )
        networks[gid] = net_suid

    print(f"[BUILD] Created {len(networks)} networks in Cytoscape: {networks}")
    return networks


# ======================================================================
# 6. Streaming mode: periodically update metrics / networks
# ======================================================================

def main_stream(interval_sec: int,
                sql_files: str,
                metrics_graph_id: str,
                metrics_mode: str | None = None,
                n_jobs: int | None = None,
                collection: str = "Trust Networks") -> None:
    """
    Streaming mode:

      - First iteration:
          * Load all edge layers
          * Compute shared metrics
          * Create one Cytoscape network per layer

      - Each subsequent iteration:
          * Reload edge layers
          * Recompute metrics (on metrics_graph_id)
          * For each graph_id:
              - If edges unchanged: update node metrics only
              - If edges changed:
                    delete old network, recreate new network
              - Handle added/removed graph_ids gracefully
    """
    print(f"[STREAM] Starting streaming mode (interval={interval_sec}s, metrics_graph_id={metrics_graph_id})")

    # INITIAL BUILD
    edge_layers = load_edge_layers_from_sql(sql_files)
    df_metrics_all = compute_metrics_for_shared_avatars(
        edge_layers=edge_layers,
        metrics_graph_id=metrics_graph_id,
        metrics_mode=metrics_mode,
        n_jobs=n_jobs
    )

    networks: dict[str, int] = {}
    for gid, df_edges in edge_layers.items():
        net_suid = create_cytoscape_network_for_layer(
            graph_id=gid,
            df_edges=df_edges,
            df_metrics_all=df_metrics_all,
            collection=collection
        )
        networks[gid] = net_suid

    edge_layers_current = {gid: df.copy() for gid, df in edge_layers.items()}

    # STREAM LOOP
    while True:
        print(f"[STREAM] Sleeping {interval_sec}s...")
        time.sleep(interval_sec)

        print("[STREAM] Reloading edge layers & recomputing metrics...")
        new_edge_layers = load_edge_layers_from_sql(sql_files)
        new_df_metrics_all = compute_metrics_for_shared_avatars(
            edge_layers=new_edge_layers,
            metrics_graph_id=metrics_graph_id,
            metrics_mode=metrics_mode,
            n_jobs=n_jobs
        )

        # Graph IDs present in new data and/or old
        all_graph_ids = set(edge_layers_current.keys()) | set(new_edge_layers.keys())

        for gid in all_graph_ids:
            old_edges = edge_layers_current.get(gid)
            new_edges = new_edge_layers.get(gid)
            old_net = networks.get(gid)

            if new_edges is not None and old_edges is not None:
                # Graph exists in both old and new
                if same_edge_set(old_edges, new_edges):
                    # Only metrics changed -> update node metrics
                    if old_net is not None:
                        print(f"[STREAM] {gid}: edges unchanged -> updating node metrics only")
                        update_node_metrics_only(old_net, new_df_metrics_all)
                else:
                    # Edges changed -> recreate network
                    print(f"[STREAM] {gid}: edges changed -> recreating network")
                    try:
                        if old_net is not None:
                            p4c.delete_network(old_net)
                            print(f"[STREAM]   Deleted old network SUID={old_net}")
                    except Exception as e:
                        print(f"[STREAM]   Warning: failed to delete old network {old_net}: {e}")

                    new_net = create_cytoscape_network_for_layer(
                        graph_id=gid,
                        df_edges=new_edges,
                        df_metrics_all=new_df_metrics_all,
                        collection=collection
                    )
                    networks[gid] = new_net

            elif new_edges is not None and old_edges is None:
                # New graph appeared -> create network
                print(f"[STREAM] {gid}: NEW graph -> creating network")
                new_net = create_cytoscape_network_for_layer(
                    graph_id=gid,
                    df_edges=new_edges,
                    df_metrics_all=new_df_metrics_all,
                    collection=collection
                )
                networks[gid] = new_net

            elif new_edges is None and old_edges is not None:
                # Graph removed -> delete network
                print(f"[STREAM] {gid}: graph REMOVED -> deleting network")
                try:
                    if old_net is not None:
                        p4c.delete_network(old_net)
                        print(f"[STREAM]   Deleted network SUID={old_net}")
                except Exception as e:
                    print(f"[STREAM]   Warning: failed to delete network {old_net}: {e}")
                networks.pop(gid, None)

        # Update current state
        edge_layers_current = {gid: df.copy() for gid, df in new_edge_layers.items()}
        df_metrics_all = new_df_metrics_all


# ======================================================================
# 7. CLI wrapper
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stream multiple trust graphs and shared metrics into Cytoscape")

    parser.add_argument(
        "--run-type",
        choices=["once", "stream"],
        default="once",
        help="Run once or run in streaming mode"
    )

    parser.add_argument(
        "--sql-files",
        type=str,
        default=None,
        help="Comma-separated list of .sql files defining graphs, e.g. "
             "sql/crc_v2_trust.sql,sql/crc_v2_invites.sql"
    )

    parser.add_argument(
        "--metrics-graph-id",
        type=str,
        default=None,
        help="graph_id (basename of SQL file) to use for metrics, e.g. crc_v2_trust. "
             "If omitted, first SQL file's basename is used."
    )

    parser.add_argument(
        "--metrics-mode",
        type=str,
        default=None,
        help="Metrics mode for GraphMetrics (e.g. basic, essential, topology,community, all). "
             "If omitted, GraphMetrics defaults to 'all'."
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between updates in streaming mode (default: 300)"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for GraphMetrics (default: auto)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="Trust Networks",
        help="Cytoscape network collection name"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # sql_files: from CLI or env
    sql_files = args.sql_files or os.getenv("GRAPH_SQL_FILES")
    if not sql_files:
        raise ValueError("You must provide --sql-files or set GRAPH_SQL_FILES in .env")

    # infer metrics_graph_id if not provided
    metrics_graph_id = args.metrics_graph_id
    if metrics_graph_id is None:
        first_sql = sql_files.split(",")[0].strip()
        metrics_graph_id = os.path.splitext(os.path.basename(first_sql))[0]
        print(f"[MAIN] metrics_graph_id not provided -> using '{metrics_graph_id}'")

    # n_jobs: CLI or env or None
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        n_jobs_env = os.getenv("N_JOBS")
        n_jobs = int(n_jobs_env) if n_jobs_env else None

    if args.run_type == "once":
        build_networks_with_shared_metrics(
            sql_files=sql_files,
            metrics_graph_id=metrics_graph_id,
            metrics_mode=args.metrics_mode,
            n_jobs=n_jobs,
            collection=args.collection
        )
    else:
        main_stream(
            interval_sec=args.interval,
            sql_files=sql_files,
            metrics_graph_id=metrics_graph_id,
            metrics_mode=args.metrics_mode,
            n_jobs=n_jobs,
            collection=args.collection
        )


if __name__ == "__main__":
    main()
