"""
Network Service

Main service for managing network/graph data including:
- Loading from SQL
- Metrics computation
- Layout management
- Graph element access
"""

import time
from datetime import datetime
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
        self.node_properties_dfs: Dict[str, pd.DataFrame] = {}  # version -> properties DataFrame
        self.layouts: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.graphs: Dict[str, nx.DiGraph] = {}
        
        # Track loaded properties
        self._loaded_property_names: List[str] = []
        self._properties_source: Optional[str] = None
        
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
    def available_node_properties_files(self) -> List[Dict[str, str]]:
        """Get list of available node properties SQL files."""
        properties_files = []
        properties_dir = settings.NODE_PROPERTIES_DIR
        if properties_dir.exists():
            for sql_path in properties_dir.glob("*.sql"):
                properties_files.append({
                    "filename": sql_path.name,
                    "name": sql_path.stem,
                    "path": str(sql_path)
                })
        return properties_files
    
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

    def load_node_properties_from_sql(
        self, 
        properties_files: List[str],
        skip_sql: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Load node properties from SQL files or cache.
        
        Expects SQL to return rows with 'avatar' column and property columns.
        Property columns can be any type including boolean.
        
        Args:
            properties_files: List of SQL file names in the properties directory
            skip_sql: If True, try to load from cache only
            
        Returns:
            Dictionary of DataFrames keyed by version extracted from filename
        """
        properties_by_version: Dict[str, pd.DataFrame] = {}
        
        for sql_file in properties_files:
            sql_path = settings.NODE_PROPERTIES_DIR / sql_file
            version = self._extract_version(sql_path.stem)
            
            # Try loading from cache if skip_sql
            if skip_sql:
                cached_df = self.cache_service.load_properties_cache(version)
                if cached_df is not None:
                    properties_by_version[version] = cached_df
                    continue
                else:
                    print(f"[PROPERTIES] No cache found for {version}, will query SQL")
            
            if not sql_path.exists():
                print(f"[PROPERTIES] File not found: {sql_path}")
                continue
            
            print(f"[PROPERTIES] Executing: {sql_file}")
            start_time = time.time()
            
            try:
                engine = self._get_db_engine()
                
                with open(sql_path, 'r', encoding='utf-8') as f:
                    sql_query = f.read()
                
                with engine.connect() as conn:
                    df = pd.read_sql(text(sql_query), conn)
                
                # Ensure avatar column exists
                if 'avatar' not in df.columns:
                    print(f"[PROPERTIES] Warning: No 'avatar' column in {sql_file}")
                    continue
                
                # Normalize avatar column to lowercase
                df['avatar'] = df['avatar'].astype(str).str.lower()
                
                # Merge if version already exists
                if version in properties_by_version:
                    existing_df = properties_by_version[version]
                    df = existing_df.merge(df, on='avatar', how='outer')
                
                properties_by_version[version] = df
                
                # Save to cache
                self.cache_service.save_properties_cache(df, version)
                
                print(f"[PROPERTIES] Loaded {len(df)} rows with "
                      f"{len(df.columns)-1} properties from {sql_file} "
                      f"in {time.time() - start_time:.2f}s")
                print(f"[PROPERTIES] Unique avatars: {df['avatar'].nunique()}, Total: {len(df)}")
                
            except Exception as e:
                print(f"[PROPERTIES] Error executing {sql_file}: {e}")
                import traceback
                traceback.print_exc()
        
        return properties_by_version
    
    def _merge_properties_to_graph(
        self,
        G: nx.DiGraph,
        properties_df: pd.DataFrame,
        version: str
    ) -> List[str]:
        """
        Merge properties DataFrame into graph node attributes.
        
        Args:
            G: NetworkX graph to update
            properties_df: DataFrame with 'avatar' and property columns
            version: Version identifier for logging
            
        Returns:
            List of property names added
        """
        if properties_df is None or properties_df.empty:
            return []
        
        property_columns = [c for c in properties_df.columns if c != 'avatar']
        if not property_columns:
            return []
        
        print(f"[PROPERTIES] DataFrame shape: {properties_df.shape}")
        print(f"[PROPERTIES] Unique avatars: {properties_df['avatar'].nunique()}")
        print(f"[PROPERTIES] Total rows: {len(properties_df)}")
        
        # Check for duplicates
        dup_mask = properties_df['avatar'].duplicated(keep=False)
        if dup_mask.any():
            dup_count = properties_df['avatar'].duplicated().sum()
            dup_avatars = properties_df.loc[dup_mask, 'avatar'].unique()[:5]
            print(f"[PROPERTIES] Warning: {dup_count} duplicate avatars found")
            print(f"[PROPERTIES] Sample duplicates: {list(dup_avatars)}")
            properties_df = properties_df.drop_duplicates(subset='avatar', keep='last')
        
        # Convert to dict for fast lookup
        properties_dict = properties_df.set_index('avatar').to_dict('index')
        
        updated_count = 0
        for node in G.nodes():
            node_str = str(node).lower()
            if node_str in properties_dict:
                props = properties_dict[node_str]
                for prop_name, value in props.items():
                    sanitized = self._sanitize_property_value(value)
                    G.nodes[node][prop_name] = sanitized
                updated_count += 1
        
        print(f"[PROPERTIES] Merged {len(property_columns)} properties "
              f"to {updated_count} nodes in {version}")
        
        return property_columns
    
    def _sanitize_property_value(self, value: Any) -> Any:
        """
        Sanitize property values for JSON serialization.
        
        Handles booleans, NaN, numpy types, arrays, and other edge cases.
        
        Args:
            value: Raw property value
            
        Returns:
            JSON-safe value
        """
        if value is None:
            return None
        
        # Handle arrays/lists BEFORE pd.isna() to avoid ambiguous truth value error
        if isinstance(value, np.ndarray):
            return [self._sanitize_property_value(v) for v in value.tolist()]
        
        if isinstance(value, (list, tuple)):
            return [self._sanitize_property_value(v) for v in value]
        
        # Handle pandas NA/NaN (only for scalar values)
        try:
            if pd.isna(value):
                return None
        except (ValueError, TypeError):
            # pd.isna() can fail on some types, continue processing
            pass
        
        # Handle numpy types
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value) if not np.isnan(value) else None
        
        # Handle datetime
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        
        # Handle booleans (Python native)
        if isinstance(value, bool):
            return value
        
        # Handle strings
        if isinstance(value, str):
            return value
        
        # Handle numeric types
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return None
            return value
        
        # Fallback: convert to string
        return str(value)
    
    def get_node_updates(
        self, 
        graph_id: str, 
        node_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current node data for incremental frontend updates.
        
        Used by auto-reload to refresh frontend display without full reload.
        
        Args:
            graph_id: Graph identifier
            node_ids: Optional list of specific node IDs (None = all nodes)
            
        Returns:
            List of node data dictionaries
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph not found: {graph_id}")
        
        G = self.graphs[graph_id]
        layout = self.layouts.get(graph_id, {})
        
        updates = []
        nodes_to_process = node_ids if node_ids else list(G.nodes())
        
        for node in nodes_to_process:
            if not G.has_node(node):
                continue
            
            node_data = {'id': str(node)}
            
            # Add all node attributes
            for key, value in G.nodes[node].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    node_data[key] = float(value) if pd.notna(value) else 0.0
                elif isinstance(value, (bool, np.bool_)):
                    node_data[key] = bool(value)
                elif isinstance(value, str):
                    node_data[key] = value
                elif value is None:
                    node_data[key] = None
                else:
                    node_data[key] = str(value)
            
            # Add position if available
            pos = layout.get(str(node))
            if pos:
                node_data['position'] = pos
            
            updates.append(clean_numpy_types(node_data))
        
        return updates
    
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
        
        Four-phase loading:
        1. Load edge data & compute metrics (per version)
        2. Load node properties (if configured)
        3. Build graphs & compute layouts
        4. Atomic state swap
        
        Args:
            config: Load configuration
            
        Returns:
            NetworkState with load results
        """
        start_time = time.time()
        
        # Phase 1: Load edge data
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
        
        # Phase 1.5: Load node properties (if configured)
        node_properties: Dict[str, pd.DataFrame] = {}
        property_names: List[str] = []
        properties_source: Optional[str] = None
        
        if config.node_properties_files:
            node_properties = self.load_node_properties_from_sql(
                config.node_properties_files,
                skip_sql=config.skip_sql
            )
            properties_source = "cache" if config.skip_sql else "sql"
            # Collect all property names
            for df in node_properties.values():
                for col in df.columns:
                    if col != 'avatar' and col not in property_names:
                        property_names.append(col)
        
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
            
            # Add node properties as node attributes
            if version in node_properties:
                self._merge_properties_to_graph(G, node_properties[version], version)
            
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
        self.node_properties_dfs = node_properties
        self.graphs = graphs
        self.layouts = layouts
        self._loaded_property_names = property_names
        self._properties_source = properties_source
        
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
            data_source=data_source,
            node_properties_loaded=property_names,
            node_properties_source=properties_source
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
    
    def get_all_node_data_df(self, graph_id: str = None) -> Optional[pd.DataFrame]:
        """
        Get DataFrame with ALL node attributes (metrics + properties).
        
        This builds a DataFrame from the graph nodes which contain both
        computed metrics and loaded properties.
        
        Args:
            graph_id: Specific graph to use, or None for first available
            
        Returns:
            DataFrame with all node attributes or None
        """
        # Get target graph
        if graph_id and graph_id in self.graphs:
            G = self.graphs[graph_id]
        elif self.graphs:
            G = list(self.graphs.values())[0]
        else:
            return None
        
        # Build DataFrame from node attributes
        rows = []
        for node_id in G.nodes():
            row = {'avatar': str(node_id)}
            for key, value in G.nodes[node_id].items():
                # Skip non-scalar values (arrays, objects)
                if isinstance(value, (list, dict, np.ndarray)):
                    continue
                row[key] = value
            rows.append(row)
        
        if not rows:
            return None
        
        return pd.DataFrame(rows)
    
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