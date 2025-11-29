"""
Network Service

Main service for managing network/graph data including:
- Loading from SQL
- Metrics computation
- Layout management
- Graph element access
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import networkx as nx
from sqlalchemy import create_engine, text

from ..config import settings, HAS_ANOMALY, HAS_SSE
from ..models.requests import LoadConfig, MetricsConfig
from ..models.responses import NetworkState
from .cache_service import CacheService
from .layout_service import LayoutService
from .auto_reload_service import AutoReloadManager
from ..utils.helpers import clean_numpy_types

from engines.graph_metrics  import GraphMetrics, METRIC_PRESETS

if HAS_ANOMALY:
    from engines.anomaly_engine import AnomalyEngine
    from engines.composite_engine import CompositeMetricEngine


class NetworkService:
    """
    Main service for network/graph management.
    
    Manages:
    - Edge layer data (DataFrames)
    - Node metrics (DataFrames, per version)
    - Graph structures (NetworkX DiGraphs)
    - Layouts (positions dictionaries)
    """
    
    def __init__(self):
        """Initialize network service."""
        # Data storage
        self.edge_layers: Dict[str, pd.DataFrame] = {}
        self.metrics_dfs: Dict[str, pd.DataFrame] = {}  # version -> DataFrame
        self.layouts: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.graphs: Dict[str, nx.DiGraph] = {}
        
        # Services
        self.cache_service = CacheService()
        self.layout_service = LayoutService()
        self.auto_reload_manager = AutoReloadManager()
        self.auto_reload_manager.set_network_service(self)
        
        # Anomaly detection (if available)
        self.anomaly_engine = AnomalyEngine() if HAS_ANOMALY else None
        self.composite_engine = CompositeMetricEngine() if HAS_ANOMALY else None
        
        # SQL files
        self._available_sql_files: Optional[List[str]] = None
        
        # Database engine (lazy init)
        self._db_engine = None
    
    @property
    def available_sql_files(self) -> List[Dict[str, str]]:
        """Get list of available SQL files with metadata."""
        if self._available_sql_files is None:
            self._available_sql_files = []
            if settings.SQL_DIR.exists():
                for sql_path in settings.SQL_DIR.glob("*.sql"):
                    self._available_sql_files.append({
                        "filename": sql_path.name,
                        "graph_id": sql_path.stem,
                        "path": str(sql_path)
                    })
        return self._available_sql_files
    
    @property
    def cytoscape_available(self) -> bool:
        """Check if Cytoscape Desktop is available."""
        return self.layout_service.cytoscape_available
    
    def _get_db_engine(self):
        """Get or create database engine."""
        if self._db_engine is None:
            self._db_engine = create_engine(settings.database_url)
        return self._db_engine
    
    def _extract_version(self, graph_id: str) -> str:
        """Extract version from graph_id (e.g., 'crc_v2_trusts' -> 'v2')."""
        parts = graph_id.lower().split('_')
        for part in parts:
            if part.startswith('v') and part[1:].isdigit():
                return part
        return "default"
    
    def load_edge_layers_from_sql(self, sql_files: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load edge layers from SQL files.
        
        Args:
            sql_files: List of SQL file names to execute
            
        Returns:
            Dictionary of DataFrames keyed by file name (without extension)
        """
        engine = self._get_db_engine()
        edge_layers = {}
        
        for sql_file in sql_files:
            sql_path = settings.SQL_DIR / sql_file
            if not sql_path.exists():
                print(f"[SQL] File not found: {sql_path}")
                continue
            
            print(f"[SQL] Executing: {sql_file}")
            start_time = time.time()
            
            try:
                with open(sql_path, 'r') as f:
                    sql_query = f.read()
                
                with engine.connect() as conn:
                    df = pd.read_sql(text(sql_query), conn)
                
                # Use filename (without extension) as key
                layer_id = sql_path.stem
                edge_layers[layer_id] = df
                
                print(f"[SQL] Loaded {len(df)} rows from {sql_file} in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                print(f"[SQL] Error executing {sql_file}: {e}")
        
        return edge_layers
    
    def compute_metrics_for_shared_avatars(
        self,
        edge_layers: Dict[str, pd.DataFrame],
        metrics_mode: str = "essential"
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute metrics for each version's shared avatars.
        
        Args:
            edge_layers: Edge layer DataFrames
            metrics_mode: Metrics computation mode
            
        Returns:
            Dictionary of metrics DataFrames keyed by version
        """
        # Group layers by version
        version_layers: Dict[str, Dict[str, pd.DataFrame]] = {}
        
        for layer_id, df in edge_layers.items():
            version = self._extract_version(layer_id)
            if version not in version_layers:
                version_layers[version] = {}
            version_layers[version][layer_id] = df
        
        # Compute metrics for each version
        metrics_dfs = {}
        
        for version, layers in version_layers.items():
            print(f"[METRICS] Computing for version: {version}")
            
            # Collect all avatars from this version
            all_avatars = set()
            for df in layers.values():
                # Check for source/target columns first
                if 'source' in df.columns:
                    all_avatars.update(df['source'].unique())
                if 'target' in df.columns:
                    all_avatars.update(df['target'].unique())
                # Fallback to other column names
                if 'truster' in df.columns:
                    all_avatars.update(df['truster'].unique())
                if 'trustee' in df.columns:
                    all_avatars.update(df['trustee'].unique())
                if 'sender' in df.columns:
                    all_avatars.update(df['sender'].unique())
                if 'receiver' in df.columns:
                    all_avatars.update(df['receiver'].unique())
            
            if not all_avatars:
                print(f"[METRICS] No avatars found for version {version}")
                continue
            
            # Build combined graph for metrics
            G = nx.DiGraph()
            for layer_id, df in layers.items():
                # Determine source/target columns
                if 'source' in df.columns and 'target' in df.columns:
                    src_col, tgt_col = 'source', 'target'
                elif 'truster' in df.columns and 'trustee' in df.columns:
                    src_col, tgt_col = 'truster', 'trustee'
                elif 'sender' in df.columns and 'receiver' in df.columns:
                    src_col, tgt_col = 'sender', 'receiver'
                else:
                    continue
                
                # Add edges efficiently
                G.add_edges_from(df[[src_col, tgt_col]].itertuples(index=False, name=None))
            
            # Compute metrics - GraphMetrics takes metrics_mode in __init__
            gm = GraphMetrics(G, n_jobs=settings.N_JOBS, metrics_mode=metrics_mode)
            
            # compute_all() returns DataFrame with 'avatar' column already set
            metrics_df = gm.compute_all()
            
            metrics_dfs[version] = metrics_df
            print(f"[METRICS] Computed {len(metrics_df.columns)-1} metrics for {len(metrics_df)} nodes in {version}")
        
        return metrics_dfs
    
    def load_network(self, config: LoadConfig) -> NetworkState:
        """
        Load network data with optional caching.
        
        Three-phase loading:
        1. Load data & compute metrics (per version)
        2. Build graphs & compute layouts
        3. Atomic state swap
        
        Args:
            config: Load configuration
            
        Returns:
            NetworkState with load results
        """
        start_time = time.time()
        
        # Phase 1: Load data
        if config.skip_sql:
            # Try loading from cache
            edge_layers = {}
            for sql_file in config.sql_files:
                layer_id = Path(sql_file).stem
                cached_df = self.cache_service.load_data_cache(layer_id)
                if cached_df is not None:
                    edge_layers[layer_id] = cached_df
            data_source = "cache"
        else:
            # Load from SQL
            edge_layers = self.load_edge_layers_from_sql(config.sql_files)
            data_source = "sql"
            
            # Save to cache
            for layer_id, df in edge_layers.items():
                self.cache_service.save_data_cache(layer_id, df)
        
        if not edge_layers:
            raise ValueError("No data loaded")
        
        # Compute metrics
        metrics_dfs = self.compute_metrics_for_shared_avatars(
            edge_layers, 
            config.metrics_mode
        )
        
        # Save metrics to cache
        for version, df in metrics_dfs.items():
            self.cache_service.save_metrics_cache(df, version)
        
        # Phase 2: Build graphs and layouts
        graphs = {}
        layouts = {}
        layout_time = 0.0
        layout_algorithm = "unknown"
        layout_cached = False
        
        for layer_id, df in edge_layers.items():
            # Build graph
            G = nx.DiGraph()
            
            # Determine source/target columns
            if 'source' in df.columns and 'target' in df.columns:
                src_col, tgt_col = 'source', 'target'
            elif 'truster' in df.columns and 'trustee' in df.columns:
                src_col, tgt_col = 'truster', 'trustee'
            elif 'sender' in df.columns and 'receiver' in df.columns:
                src_col, tgt_col = 'sender', 'receiver'
            else:
                print(f"[GRAPH] Unknown edge columns in {layer_id}: {list(df.columns)}")
                continue
            
            # Add edges efficiently
            G.add_edges_from(df[[src_col, tgt_col]].itertuples(index=False, name=None))
            
            # Add metrics as node attributes
            version = self._extract_version(layer_id)
            if version in metrics_dfs:
                metrics_df = metrics_dfs[version]
                metrics_dict = metrics_df.set_index('avatar').to_dict('index')
                
                for node in G.nodes():
                    if node in metrics_dict:
                        for metric, value in metrics_dict[node].items():
                            G.nodes[node][metric] = value
            
            graphs[layer_id] = G
            
            # Get layout
            cached_layout = None
            if config.use_cached_layout:
                cached_layout = self.cache_service.get_cached_layout(
                    layer_id, G.number_of_nodes(), G.number_of_edges()
                )
                if cached_layout:
                    layout_cached = True
            
            positions, algorithm, comp_time = self.layout_service.compute_layout(
                G, layer_id, cached_layout
            )
            
            layouts[layer_id] = positions
            layout_time += comp_time
            layout_algorithm = algorithm
            
            # Save layout to cache
            if algorithm != "cached":
                self.cache_service.save_layout_cache(
                    layer_id,
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    positions,
                    {'algorithm': algorithm}
                )
        
        # Phase 3: Atomic state swap
        self.edge_layers = edge_layers
        self.metrics_dfs = metrics_dfs
        self.graphs = graphs
        self.layouts = layouts
        
        # Compute totals
        total_nodes = sum(G.number_of_nodes() for G in graphs.values())
        total_edges = sum(G.number_of_edges() for G in graphs.values())
        
        # Get metric names
        metric_names = []
        for df in metrics_dfs.values():
            metric_names.extend([c for c in df.columns if c != 'avatar'])
            break  # All versions have same metrics
        
        return NetworkState(
            loaded_graphs=list(graphs.keys()),
            node_count=total_nodes,
            edge_count=total_edges,
            metrics_computed=metric_names,
            computation_time=time.time() - start_time,
            layout_computation_time=layout_time,
            layout_algorithm=layout_algorithm,
            layout_cached=layout_cached,
            data_source=data_source
        )
    
    def update_metrics(self, config: MetricsConfig) -> Dict[str, Any]:
        """
        Re-run metrics on existing graphs and update node attributes.
        Only updates metrics for the VERSION matching the target graph.
        """
        if not self.edge_layers:
            raise ValueError("No graphs loaded. Please load networks first.")

        target_graph = config.metrics_graph_id
        if not target_graph:
            # Default to first available if not specified
            target_graph = list(self.edge_layers.keys())[0]

        target_version = self._extract_version(target_graph)
        print(f"[METRICS] Updating metrics for version: {target_version} (Target: {target_graph})")
        
        start_time = time.time()
        
        new_metrics_df = self.compute_metrics_for_shared_avatars(
            edge_layers=self.edge_layers,
            metrics_mode=config.metrics_mode
        )
        
        # Get the metrics for target version
        if target_version not in new_metrics_df:
            raise ValueError(f"No metrics computed for version {target_version}")
        
        metrics_df = new_metrics_df[target_version]
        
        # Update state for this version
        self.metrics_dfs[target_version] = metrics_df
        
        # Cache new metrics for this version
        self.cache_service.save_metrics_cache(metrics_df, target_version)
        
        # Update graph objects in memory (ONLY graphs of this version)
        metrics_dict = metrics_df.set_index('avatar').to_dict('index')
        node_updates = []
        
        for avatar, attrs in metrics_dict.items():
            clean_attrs = {
                k: (int(v) if isinstance(v, (np.int64, np.int32)) else 
                    float(v) if isinstance(v, (np.float64, np.float32)) else v) 
                for k, v in attrs.items()
            }
            
            # Update NetworkX graphs matching this version
            for gid, G in self.graphs.items():
                if self._extract_version(gid) == target_version and G.has_node(avatar):
                    for k, v in clean_attrs.items():
                        G.nodes[avatar][k] = v
            
            clean_attrs['id'] = avatar
            node_updates.append(clean_attrs)

        elapsed = time.time() - start_time
        print(f"[METRICS] Updated {len(node_updates)} nodes in {elapsed:.2f}s")
        
        return {
            "metrics_computed": list(metrics_df.columns),
            "computation_time": elapsed,
            "node_data": node_updates
        }
    
    def get_graph_elements(
        self, 
        graph_id: str, 
        mode: str = "full"
    ) -> List[Dict[str, Any]]:
        """
        Get graph elements for Cytoscape.js.
        
        Args:
            graph_id: Graph identifier
            mode: "full" for nodes+edges, "nodes_only" for just nodes
            
        Returns:
            List of Cytoscape.js elements
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        
        G = self.graphs[graph_id]
        layout = self.layouts.get(graph_id, {})
        
        elements = []
        
        # Add nodes
        for node in G.nodes():
            node_data = {'id': str(node)}
            
            # Add metrics
            for key, value in G.nodes[node].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    node_data[key] = float(value) if pd.notna(value) else 0.0
                elif isinstance(value, str):
                    node_data[key] = value
            
            # Add position
            pos = layout.get(str(node), {'x': 0, 'y': 0})
            
            elements.append({
                'group': 'nodes',
                'data': clean_numpy_types(node_data),
                'position': pos
            })
        
        # Add edges (if not nodes_only)
        if mode != "nodes_only":
            for u, v in G.edges():
                elements.append({
                    'group': 'edges',
                    'data': {
                        'id': f"{u}-{v}",
                        'source': str(u),
                        'target': str(v)
                    }
                })
        
        return elements
    
    def get_graph_edges_chunk(
        self, 
        graph_id: str, 
        offset: int = 0, 
        limit: int = 50000
    ) -> Dict[str, Any]:
        """
        Get a chunk of edges for incremental loading.
        
        Args:
            graph_id: Graph identifier
            offset: Starting edge index
            limit: Maximum edges to return
            
        Returns:
            Dictionary with edges and pagination info
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        
        G = self.graphs[graph_id]
        edges = list(G.edges())
        total = len(edges)
        
        # Get chunk
        chunk_edges = edges[offset:offset + limit]
        
        elements = []
        for u, v in chunk_edges:
            elements.append({
                'group': 'edges',
                'data': {
                    'id': f"{u}-{v}",
                    'source': str(u),
                    'target': str(v)
                }
            })
        
        return {
            'edges': elements,
            'offset': offset,
            'limit': limit,
            'returned': len(elements),
            'total': total,
            'has_more': offset + len(elements) < total
        }
    
    def get_metrics_dataframe(self, version: str = None) -> Optional[pd.DataFrame]:
        """
        Get metrics DataFrame for a version.
        
        Args:
            version: Version identifier (None = first available)
            
        Returns:
            Metrics DataFrame or None
        """
        if version and version in self.metrics_dfs:
            return self.metrics_dfs[version]
        elif self.metrics_dfs:
            return list(self.metrics_dfs.values())[0]
        return None
    
    def list_cached_layouts(self) -> List[Dict[str, Any]]:
        """List all cached layouts."""
        return self.cache_service.list_cached_layouts()
    
    def clear_layout_cache(self, graph_id: Optional[str] = None):
        """Clear layout cache."""
        self.cache_service.clear_layout_cache(graph_id)

    def get_current_metrics_df(self) -> Optional[pd.DataFrame]:
        """
        Get the current metrics DataFrame.
        
        Alias for get_metrics_dataframe() for compatibility with anomaly router.
        
        Returns:
            Metrics DataFrame or None if no data loaded
        """
        return self.get_metrics_dataframe()
    
    def update_node_data(self, node_updates: List[Dict[str, Any]]) -> None:
        """
        Update node attributes in all graphs.
        
        Used by anomaly detection to apply scores to graph nodes.
        
        Args:
            node_updates: List of dicts with 'id' and attribute key-value pairs
        """
        for update in node_updates:
            node_id = update.get('id')
            if not node_id:
                continue
            
            # Update in all graphs
            for gid, G in self.graphs.items():
                if G.has_node(node_id):
                    for key, value in update.items():
                        if key != 'id':
                            G.nodes[node_id][key] = value

    def get_neighbors(self, graph_id: str, node_ids: List[str], direction: str = "both") -> Dict[str, Any]:
        """
        Get neighbors of specified nodes from the graph.
        
        Args:
            graph_id: The graph to query
            node_ids: List of node IDs to find neighbors for
            direction: "in" for predecessors, "out" for successors, "both" for all
            
        Returns:
            Dict with incoming, outgoing neighbor lists and counts
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not found")
        
        G = self.graphs[graph_id]
        
        incoming_set = set()
        outgoing_set = set()
        
        for node_id in node_ids:
            if not G.has_node(node_id):
                continue
            
            if direction in ("in", "both"):
                incoming_set.update(G.predecessors(node_id))
            
            if direction in ("out", "both"):
                outgoing_set.update(G.successors(node_id))
        
        # Remove the source nodes from results
        source_set = set(node_ids)
        incoming_set -= source_set
        outgoing_set -= source_set
        
        return {
            "incoming": list(incoming_set),
            "outgoing": list(outgoing_set),
            "incoming_count": len(incoming_set),
            "outgoing_count": len(outgoing_set),
            "source_nodes": node_ids,
        }

# Singleton instance
network_service = NetworkService()