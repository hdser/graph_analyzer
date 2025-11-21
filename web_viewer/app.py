"""
Web viewer with Cytoscape Desktop layout computation and caching
Supports Server-Side Incremental Layout Updates
FIX: Changed async routes to sync (def) to prevent blocking the event loop.
FIX: Metrics caching is now version-aware (v1, v2) to prevent cross-contamination.
"""

import os
import json
import time
import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

import sys
sys.path.append(str(Path(__file__).parent.parent))
from graph_metrics import GraphMetrics, METRIC_CATEGORIES, METRIC_PRESETS

# Try to import py4cytoscape for Cytoscape Desktop support
try:
    import py4cytoscape as p4c
    HAS_CYTOSCAPE_DESKTOP = True
    try:
        p4c.cytoscape_ping()
        print("[CYTOSCAPE] Cytoscape Desktop is available for layout computation")
    except:
        print("[CYTOSCAPE] Cytoscape Desktop is installed but not running")
        HAS_CYTOSCAPE_DESKTOP = False
except ImportError:
    HAS_CYTOSCAPE_DESKTOP = False
    print("[CYTOSCAPE] py4cytoscape not installed, will use layout service")

load_dotenv(Path(__file__).parent.parent / '.env')

print("=" * 70)
print("GRAPH ANALYZER WEB VIEWER")
if HAS_CYTOSCAPE_DESKTOP:
    print("Using Cytoscape Desktop for fast, high-quality layouts")
else:
    print("Using Cytoscape.js layout service")
print("=" * 70)

app = FastAPI(title="Graph Analyzer Web Viewer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# Data Models
# ============================================================================

class LoadConfig(BaseModel):
    sql_files: List[str]
    use_cached_layout: bool = True
    skip_sql: bool = False  # New flag to bypass SQL

class MetricsConfig(BaseModel):
    metrics_mode: str = "basic"
    metrics_graph_id: Optional[str] = None

class NetworkState(BaseModel):
    loaded_graphs: List[str]
    current_graph: Optional[str]
    node_count: int
    edge_count: int
    metrics_computed: List[str]
    computation_time: float
    layout_computation_time: float
    layout_algorithm: str
    layout_cached: bool
    data_source: str  # 'sql' or 'cache'


# ============================================================================
# Network Service
# ============================================================================

class NetworkService:
    """
    Network service with Cytoscape Desktop layout support, caching, and
    Incremental Layout capabilities.
    """
    
    def __init__(self):
        self.db_engine = self._create_db_engine()
        self.sql_dir = Path(__file__).parent.parent / "sql"
        self.cache_dir = Path(__file__).parent / "cache"
        self.layouts_dir = self.cache_dir / "layouts"
        self.data_cache_dir = self.cache_dir / "data"  # New data cache directory
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.layouts_dir.mkdir(exist_ok=True)
        self.data_cache_dir.mkdir(exist_ok=True)
        
        self.edge_layers = {}
        # Metrics are now stored per version to allow mixed loading without conflict
        self.metrics_dfs = {} # type: Dict[str, pd.DataFrame]
        self.layouts = {}
        self.graphs = {}
        self.current_load_config = None
        self.current_metrics_config = None
        
        self.available_sql_files = self._scan_sql_files()
        self.layout_service_url = os.getenv("LAYOUT_SERVICE_URL", "http://localhost:3001")
        
        # Check if Cytoscape Desktop is available
        self.cytoscape_available = self._check_cytoscape_desktop()
    
    def _check_cytoscape_desktop(self) -> bool:
        """Check if Cytoscape Desktop is available"""
        if not HAS_CYTOSCAPE_DESKTOP:
            return False
        try:
            p4c.cytoscape_ping()
            return True
        except:
            return False
    
    def _create_db_engine(self):
        """Create PostgreSQL connection"""
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        
        if not all([db_user, db_password, db_host, db_name]):
            # Return None if config missing, will fail only if SQL is attempted
            print("[WARNING] DB credentials missing. SQL features will fail.")
            return None
        
        url = URL.create(
            "postgresql+psycopg2",
            username=db_user,
            password=db_password,
            host=db_host,
            database=db_name,
        )
        return create_engine(url)
    
    def _scan_sql_files(self) -> List[Dict[str, str]]:
        """Scan sql directory for available SQL files"""
        sql_files = []
        if self.sql_dir.exists():
            for sql_path in self.sql_dir.glob("*.sql"):
                sql_files.append({
                    "filename": sql_path.name,
                    "graph_id": sql_path.stem,
                    "path": str(sql_path)
                })
        return sql_files
    
    def _extract_version(self, graph_id: str) -> str:
        """Extracts version string (e.g., 'v1', 'v2') from graph_id. Defaults to 'default'."""
        match = re.search(r'(v\d+)', graph_id)
        return match.group(1) if match else 'default'

    # --- Layout Caching ---
    
    def get_layout_cache_key(self, graph_id: str, node_count: int, edge_count: int) -> str:
        """Generate a cache key for a layout"""
        return f"{graph_id}_{node_count}n_{edge_count}e"
    
    def get_cached_layout(self, graph_id: str, node_count: int, edge_count: int) -> Optional[Dict]:
        """Try to get a cached layout"""
        cache_key = self.get_layout_cache_key(graph_id, node_count, edge_count)
        cache_file = self.layouts_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    print(f"[CACHE] Found cached layout for {graph_id} ({node_count} nodes, {edge_count} edges)")
                    return data['positions']
            except Exception as e:
                print(f"[CACHE] Error loading cached layout: {e}")
        
        # Try to find a similar layout (within 10% node/edge count)
        for file in self.layouts_dir.glob(f"{graph_id}_*.json"):
            try:
                parts = file.stem.split('_')
                if len(parts) >= 3:
                    cached_nodes = int(parts[-2].replace('n', ''))
                    cached_edges = int(parts[-1].replace('e', ''))
                    
                    node_diff = abs(cached_nodes - node_count) / max(cached_nodes, node_count)
                    edge_diff = abs(cached_edges - edge_count) / max(cached_edges, edge_count)
                    
                    if node_diff <= 0.1 and edge_diff <= 0.1:
                        with open(file, 'r') as f:
                            data = json.load(f)
                            print(f"[CACHE] Using similar cached layout: {cached_nodes} nodes (vs {node_count}), "
                                  f"{cached_edges} edges (vs {edge_count})")
                            return data['positions']
            except:
                continue
        
        return None
    
    def save_layout_cache(self, graph_id: str, node_count: int, edge_count: int, 
                         positions: Dict, metadata: Dict = None):
        """Save a layout to cache"""
        cache_key = self.get_layout_cache_key(graph_id, node_count, edge_count)
        cache_file = self.layouts_dir / f"{cache_key}.json"
        
        data = {
            'graph_id': graph_id,
            'node_count': node_count,
            'edge_count': edge_count,
            'positions': positions,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        print(f"[CACHE] Saved layout for {graph_id} to {cache_file.name}")

    # --- Data Caching (New) ---

    def _get_data_cache_path(self, graph_id: str, data_type: str) -> Path:
        return self.data_cache_dir / f"{graph_id}_{data_type}.csv"

    def save_data_cache(self, graph_id: str, df_edges: pd.DataFrame):
        """Save edge list to CSV cache"""
        path = self._get_data_cache_path(graph_id, 'edges')
        df_edges.to_csv(path, index=False)

    def load_data_cache(self, graph_id: str) -> Optional[pd.DataFrame]:
        """Load edge list from CSV cache"""
        path = self._get_data_cache_path(graph_id, 'edges')
        if path.exists():
            return pd.read_csv(path)
        return None

    def save_metrics_cache(self, df_metrics: pd.DataFrame, version: str):
        """Save metrics/node attributes to CSV cache, strictly by version"""
        path = self.data_cache_dir / f"node_metrics_{version}.csv"
        df_metrics.to_csv(path, index=False)
        
    def load_metrics_cache(self, version: str) -> Optional[pd.DataFrame]:
        """Load metrics/node attributes from CSV cache, strictly by version"""
        path = self.data_cache_dir / f"node_metrics_{version}.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    # --- Layout Computation ---

    def compute_layout_via_cytoscape_desktop(self, graph_id: str, df_edges: pd.DataFrame, 
                                        df_metrics_all: pd.DataFrame) -> Dict:
        """
        Use Cytoscape Desktop to compute layout via CyREST (bypassing style timeout issues)
        """
        if not self.cytoscape_available:
            raise RuntimeError("Cytoscape Desktop is not available")
        
        print(f"[LAYOUT] Using Cytoscape Desktop for {graph_id}")
        start = time.time()
        
        try:
            nodes_payload = [{"data": {"id": str(row['avatar'])}} for _, row in df_metrics_all.iterrows()]
            edges_payload = [
                {"data": {"source": str(row['source']), "target": str(row['target'])}} 
                for _, row in df_edges.iterrows()
            ]
            
            title = f"web_viewer_{graph_id}_{int(time.time())}"
            
            print(f"[LAYOUT] Creating network via CyREST (bypassing vizmap/styles)...")
            res = p4c.cyrest_post("networks", body={
                "data": {"name": title},
                "elements": {"nodes": nodes_payload, "edges": edges_payload}
            })
            net_suid = res['networkSUID']
            
            try:
                p4c.cyrest_post(f"networks/{net_suid}/views")
                time.sleep(0.2)
            except Exception as e:
                print(f"[LAYOUT] View creation note: {e}")

            print(f"[LAYOUT] Applying force-directed layout...")
            p4c.set_layout_properties(
                'force-directed-cl',
                {
                    'numIterations': 400,
                    'numIterationsEdgeRepulsive': 10,
                    'defaultSpringCoefficient': 1e-5,
                    'defaultSpringLength': 30,
                    'defaultNodeMass': 1.0,
                    'isDeterministic': True,
                    'fromScratch': True,
                    'singlePartition': False
                }
            )
            p4c.layout_network("force-directed-cl", network=net_suid)
            
            print(f"[LAYOUT] Getting positions from view...")
            views = p4c.get_network_views(net_suid)
            if not views:
                raise RuntimeError("No view found after layout")
            
            view_suid = views[0]
            view_json = p4c.cyrest_get(f"networks/{net_suid}/views/{view_suid}")
            
            positions = {}
            
            if view_json and isinstance(view_json, dict):
                elements = view_json.get('elements', {})
                nodes = elements.get('nodes', [])
                
                for node in nodes:
                    if isinstance(node, dict):
                        node_data = node.get('data', {})
                        node_position = node.get('position', {})
                        node_id = (node_data.get('name') or node_data.get('shared_name') or node_data.get('id'))
                        
                        if node_id and 'x' in node_position and 'y' in node_position:
                            positions[node_id] = {'x': float(node_position['x']), 'y': float(node_position['y'])}
            
            try:
                p4c.delete_network(net_suid)
            except:
                pass
            
            elapsed = time.time() - start
            print(f"[LAYOUT] Retrieved {len(positions)} positions in {elapsed:.2f}s")
            
            if len(positions) == 0:
                raise RuntimeError("No positions retrieved")
            
            self.save_layout_cache(
                graph_id, len(nodes_payload), len(edges_payload), positions,
                {'algorithm': 'cytoscape-desktop-force-directed', 'time': elapsed}
            )
            return positions
            
        except Exception as e:
            print(f"[LAYOUT] Error: {e}")
            try:
                if 'net_suid' in locals(): p4c.delete_network(net_suid)
            except: pass
            raise
        
    def compute_layout_via_service(self, graph_id: str, df_edges: pd.DataFrame, 
                                  algorithm: str = "fcose", locked_positions: Dict = None,
                                  initial_positions: Dict = None) -> Dict:
        """Compute layout using Cytoscape.js service."""
        import requests
        
        print(f"[LAYOUT] Computing {algorithm} layout via service for {graph_id}")
        
        if isinstance(df_edges, pd.DataFrame):
            # Standard full layout case
            edges = [{"source": row['source'], "target": row['target']} for _, row in df_edges.iterrows()]
            node_ids = list(set(df_edges['source'].tolist() + df_edges['target'].tolist()))
        elif isinstance(df_edges, list):
            # Incremental layout case (list of dicts)
            edges = df_edges
            sources = [e['source'] for e in edges]
            targets = [e['target'] for e in edges]
            node_ids = list(set(sources + targets))
        else:
            print("[LAYOUT] Error: Unknown edge format")
            return {}
        
        nodes_payload = []
        for node in node_ids:
            node_obj = {"data": {"id": node}}
            if initial_positions and node in initial_positions:
                node_obj["position"] = initial_positions[node]
            nodes_payload.append(node_obj)
        
        payload = {
            "nodes": nodes_payload,
            "edges": edges,
            "algorithm": algorithm
        }
        
        if locked_positions:
            payload["lockedPositions"] = locked_positions
        
        try:
            start = time.time()
            response = requests.post(
                f"{self.layout_service_url}/compute-layout",
                json=payload,
                timeout=300
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Layout service returned {response.status_code}")
            
            result = response.json()
            elapsed = time.time() - start
            
            if not locked_positions:
                self.save_layout_cache(
                    graph_id, len(node_ids), len(edges), result['positions'],
                    {'algorithm': algorithm, 'time': elapsed}
                )
            
            return result['positions']
            
        except Exception as e:
            print(f"[LAYOUT] Error calling layout service: {e}")
            return self.compute_circular_layout(node_ids)
    
    def compute_circular_layout(self, nodes: List) -> Dict:
        n = len(nodes)
        positions = {}
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n
            positions[node] = {"x": 1000 * np.cos(angle), "y": 1000 * np.sin(angle)}
        return positions
    
    def _calculate_centroid_positions(self, new_nodes: list, anchors: set, current_layout: dict, G: nx.DiGraph) -> dict:
        initial_positions = {}
        for node in new_nodes:
            neighbors_positions = []
            if G.has_node(node):
                for n in G.neighbors(node):
                    if n in anchors: neighbors_positions.append(current_layout[n])
                for n in G.predecessors(node):
                    if n in anchors: neighbors_positions.append(current_layout[n])
            
            if neighbors_positions:
                avg_x = sum(p['x'] for p in neighbors_positions) / len(neighbors_positions)
                avg_y = sum(p['y'] for p in neighbors_positions) / len(neighbors_positions)
                initial_positions[node] = {
                    'x': avg_x + random.uniform(-50, 50),
                    'y': avg_y + random.uniform(-50, 50)
                }
            else:
                initial_positions[node] = {
                    'x': random.uniform(-500, 500), 
                    'y': random.uniform(-500, 500)
                }
        return initial_positions

    def compute_incremental_layout(self, graph_id: str, new_nodes: list, current_layout: dict, G: nx.DiGraph):
        start_time = time.time()
        new_nodes_set = set(new_nodes)
        anchors = set()
        subgraph_edges_list = []
        
        for node in new_nodes:
            if G.has_node(node):
                for _, target in G.out_edges(node):
                    subgraph_edges_list.append({"source": node, "target": target})
                    if target not in new_nodes_set and target in current_layout:
                        anchors.add(target)
                for source, _ in G.in_edges(node):
                    subgraph_edges_list.append({"source": source, "target": node})
                    if source not in new_nodes_set and source in current_layout:
                        anchors.add(source)
        
        locked_positions = {n: current_layout[n] for n in anchors}
        initial_positions = self._calculate_centroid_positions(new_nodes, anchors, current_layout, G)
        
        # Pass list of dicts directly
        new_positions = self.compute_layout_via_service(
            graph_id, subgraph_edges_list, algorithm="fcose",
            locked_positions=locked_positions, initial_positions=initial_positions
        )
        return new_positions

    def compute_layout(self, graph_id: str, df_edges: pd.DataFrame, 
                       df_metrics_all: pd.DataFrame, use_cache: bool) -> Dict:
        """
        Compute layout with strict fallback: Cache -> Desktop -> Service (fCoSE)
        """
        node_count = len(df_metrics_all)
        edge_count = len(df_edges)
        
        # 1. Cache
        if use_cache:
            cached_layout = self.get_cached_layout(graph_id, node_count, edge_count)
            if cached_layout: 
                return cached_layout
        
        # 2. Cytoscape Desktop
        if self.cytoscape_available and edge_count < 5000000:
            try:
                return self.compute_layout_via_cytoscape_desktop(graph_id, df_edges, df_metrics_all)
            except Exception as e:
                print(f"[LAYOUT] Cytoscape Desktop failed, falling back: {e}")
        
        # 3. Service (fCoSE)
        return self.compute_layout_via_service(graph_id, df_edges, "fcose")
    
    def load_edge_layers_from_sql(self, sql_files: List[str]) -> dict:
        edge_layers = {}
        if not self.db_engine:
            raise RuntimeError("Cannot run SQL: Database engine not initialized")
            
        for filename in sql_files:
            sql_path = self.sql_dir / filename
            if not sql_path.exists(): continue
            graph_id = sql_path.stem
            with open(sql_path, 'r') as f: query = f.read()
            print(f"[SQL] Loading {filename}...")
            df = pd.read_sql_query(query, self.db_engine)
            edge_layers[graph_id] = df
        return edge_layers
    
    def compute_metrics_for_shared_avatars(self, edge_layers: dict,
                                          metrics_graph_id: str,
                                          metrics_mode: str = None) -> pd.DataFrame:
        """
        Compute metrics using ONLY edge layers that match the version of the target graph.
        """
        # 1. Determine the version of the target graph
        if metrics_graph_id not in edge_layers:
            # Fallback: use the first available graph's version
            if edge_layers:
                metrics_graph_id = list(edge_layers.keys())[0]
            else:
                return pd.DataFrame()
        
        target_version = self._extract_version(metrics_graph_id)
        print(f"[METRICS] Computing metrics for version '{target_version}' (Target: {metrics_graph_id})")

        # 2. Filter edge layers to ONLY include those of the same version
        relevant_layers = {
            gid: df for gid, df in edge_layers.items() 
            if self._extract_version(gid) == target_version
        }
        
        if metrics_graph_id not in relevant_layers:
             # Should not happen if logic above is correct, but safety first
             return pd.DataFrame()

        df_metrics_edges = relevant_layers[metrics_graph_id]
        
        # 3. Build Graph for metrics (Topology)
        G = nx.DiGraph()
        G.add_edges_from(df_metrics_edges[["source", "target"]].itertuples(index=False, name=None))
        
        # 4. Identify Universe of Avatars (ONLY for this version)
        all_avatars = set()
        for gid, df_edges in relevant_layers.items():
            all_avatars.update(df_edges["source"].unique())
            all_avatars.update(df_edges["target"].unique())
        
        # Safety check for huge graphs
        if G.number_of_nodes() > 50000:
            if not (',' in (metrics_mode or '') or metrics_mode in ["basic", "topology", "essential"]):
                metrics_mode = "basic"
        
        print(f"[METRICS] Computing metrics (Mode: {metrics_mode or 'basic'})...")
        metrics_calc = GraphMetrics(G, n_jobs=4, metrics_mode=metrics_mode or "basic")
        df_metrics = metrics_calc.compute_all()
        
        df_all = pd.DataFrame({"avatar": list(all_avatars)})
        df_metrics_all = df_all.merge(df_metrics, on="avatar", how="left")
        metric_cols = [c for c in df_metrics_all.columns if c != "avatar"]
        df_metrics_all[metric_cols] = df_metrics_all[metric_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return df_metrics_all
    
    def load_network(self, config: LoadConfig) -> NetworkState:
        """
        Load network: Loads data and computes layout. 
        Support for 'Skip SQL' via data caching.
        Separates processing by version.
        """
        start_time = time.time()
        data_source_used = "sql"
        new_edge_layers = {}
        self.metrics_dfs = {} # Reset metrics state
        
        # Group files by version
        files_by_version = {}
        for filename in config.sql_files:
            graph_id = Path(filename).stem
            ver = self._extract_version(graph_id)
            if ver not in files_by_version: files_by_version[ver] = []
            files_by_version[ver].append(filename)

        # --- PHASE 1: Load Data & Metrics (Per Version) ---
        
        for version, files in files_by_version.items():
            print(f"[LOAD] Processing version: {version}")
            version_edge_layers = {}
            
            # 1a. Load Edges
            if config.skip_sql:
                print(f"[LOAD] {version}: Attempting to load from cache...")
                all_cached = True
                for filename in files:
                    graph_id = Path(filename).stem
                    df = self.load_data_cache(graph_id)
                    if df is not None:
                        version_edge_layers[graph_id] = df
                    else:
                        print(f"[LOAD] Cache miss for {graph_id}")
                        all_cached = False
                
                # Try load metrics
                cached_metrics = self.load_metrics_cache(version)
                if all_cached and cached_metrics is not None:
                    self.metrics_dfs[version] = cached_metrics
                    new_edge_layers.update(version_edge_layers)
                    data_source_used = "cache"
                    print(f"[LOAD] {version}: Successfully loaded from cache.")
                    continue # Skip to next version
                else:
                    print(f"[LOAD] {version}: Cache incomplete. Fallback to SQL.")
                    config.skip_sql = False # Force SQL for remaining if one fails

            # 1b. SQL Fallback (or primary)
            if not config.skip_sql:
                try:
                    loaded_layers = self.load_edge_layers_from_sql(files)
                    new_edge_layers.update(loaded_layers)
                    version_edge_layers.update(loaded_layers)
                    
                    # Compute Basic Metrics for this version
                    # Use first graph in this version group as default target
                    default_target_id = Path(files[0]).stem
                    
                    metrics_df = self.compute_metrics_for_shared_avatars(
                        edge_layers=new_edge_layers, # Pass all, function filters by version of target
                        metrics_graph_id=default_target_id,
                        metrics_mode="basic" 
                    )
                    self.metrics_dfs[version] = metrics_df
                    
                    # Save Cache
                    print(f"[LOAD] {version}: Saving to cache...")
                    for gid, df in version_edge_layers.items():
                        self.save_data_cache(gid, df)
                    self.save_metrics_cache(metrics_df, version)
                    
                except Exception as e:
                    print(f"Load failed for {version}: {e}")
                    if not new_edge_layers: raise

        # --- PHASE 2: Build Graphs & Layouts ---
        
        layout_start = time.time()
        new_layouts = {}
        new_graphs = {}
        layout_cached = False
        layout_algo = "auto"
        
        # Build graphs
        for graph_id, df_edges in new_edge_layers.items():
            version = self._extract_version(graph_id)
            metrics_df = self.metrics_dfs.get(version)
            
            G = nx.DiGraph()
            
            # Add nodes (using metrics for that version)
            if metrics_df is not None:
                metrics_dict = metrics_df.set_index('avatar').to_dict('index')
                for avatar, attrs in metrics_dict.items():
                    clean_attrs = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in attrs.items()}
                    G.add_node(avatar, **clean_attrs)
            
            for _, row in df_edges.iterrows():
                G.add_edge(row['source'], row['target'])
            
            new_graphs[graph_id] = G
            
            # Compute Layout
            # Note: We pass metrics_df just to count nodes for cache key
            positions = self.compute_layout(graph_id, df_edges, metrics_df if metrics_df is not None else pd.DataFrame(), config.use_cached_layout)
            new_layouts[graph_id] = positions
            
            if config.use_cached_layout:
                cache_key = self.get_layout_cache_key(graph_id, G.number_of_nodes(), G.number_of_edges())
                if (self.layouts_dir / f"{cache_key}.json").exists():
                    layout_cached = True

        # Incremental updates pre-calc
        for graph_id in new_graphs:
            G = new_graphs[graph_id]
            layout = new_layouts.get(graph_id, {})
            existing_nodes_in_layout = set(layout.keys())
            all_nodes_in_graph = set(G.nodes())
            missing_nodes = list(all_nodes_in_graph - existing_nodes_in_layout)
            
            if missing_nodes:
                print(f"[PRE-COMPUTE] Graph {graph_id}: Finding positions for {len(missing_nodes)} missing nodes...")
                try:
                    new_positions = self.compute_incremental_layout(graph_id, missing_nodes, layout, G)
                    layout.update(new_positions)
                    new_layouts[graph_id] = layout
                    self.save_layout_cache(
                        graph_id, G.number_of_nodes(), G.number_of_edges(), layout, 
                        {'updated': True, 'update_time': datetime.now().isoformat()}
                    )
                except Exception as e:
                    print(f"[PRE-COMPUTE] Error: {e}")

        layout_time = time.time() - layout_start
        total_time = time.time() - start_time

        # --- PHASE 3: Atomic State Swap ---
        self.edge_layers = new_edge_layers
        # self.metrics_dfs is already updated
        self.graphs = new_graphs
        self.layouts = new_layouts
        self.current_load_config = config
        
        # Calculate total nodes for summary
        total_nodes = sum(len(df) for df in self.metrics_dfs.values())
        cols = []
        if self.metrics_dfs:
            cols = list(list(self.metrics_dfs.values())[0].columns)

        return NetworkState(
            loaded_graphs=list(self.graphs.keys()),
            current_graph=list(self.graphs.keys())[0] if self.graphs else None,
            node_count=total_nodes,
            edge_count=sum(len(df) for df in self.edge_layers.values()),
            metrics_computed=cols,
            computation_time=total_time,
            layout_computation_time=layout_time,
            layout_algorithm=layout_algo,
            layout_cached=layout_cached,
            data_source=data_source_used
        )

    def update_metrics(self, config: MetricsConfig) -> Dict:
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
            metrics_graph_id=target_graph,
            metrics_mode=config.metrics_mode
        )
        
        # Update state for this version
        self.metrics_dfs[target_version] = new_metrics_df
        self.current_metrics_config = config
        
        # Cache new metrics for this version
        self.save_metrics_cache(new_metrics_df, target_version)
        
        # Update graph objects in memory (ONLY graphs of this version)
        metrics_dict = new_metrics_df.set_index('avatar').to_dict('index')
        node_updates = []
        
        for avatar, attrs in metrics_dict.items():
            clean_attrs = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in attrs.items()}
            
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
            "metrics_computed": list(new_metrics_df.columns),
            "computation_time": elapsed,
            "node_data": node_updates
        }
    
    def get_graph_elements(self, graph_id: str, mode: str = "full"):
        """Return Cytoscape.js elements for a graph.

        Parameters
        ----------
        graph_id : str
            Identifier of the loaded graph.
        mode : {"full", "nodes_only"}
            If "nodes_only", only node elements (with positions) are returned.
            If "full", both nodes and edges are returned.
        """
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not loaded")

        G = self.graphs[graph_id]
        layout = self.layouts.get(graph_id, {})
        elements: list[dict] = []

        # Nodes
        for node in G.nodes():
            node_data = dict(G.nodes[node])

            # Make sure values are JSON-serialisable (convert numpy types)
            clean_data: dict[str, Any] = {}
            for k, v in node_data.items():
                if isinstance(v, (np.integer,)):
                    clean_data[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean_data[k] = float(v)
                else:
                    clean_data[k] = v

            clean_data["id"] = node
            if isinstance(node, str) and len(node) > 12:
                clean_data["label"] = node[:10] + "..."
            else:
                clean_data["label"] = node

            node_element: dict[str, Any] = {"group": "nodes", "data": clean_data}
            if node in layout:
                node_element["position"] = layout[node]
            elements.append(node_element)

        # Edges (optional)
        if mode != "nodes_only":
            for source, target in G.edges():
                edge_element = {
                    "group": "edges",
                    "data": {
                        "id": f"{source}->{target}",
                        "source": source,
                        "target": target,
                    },
                }
                elements.append(edge_element)

        return elements


    
    def list_cached_layouts(self):
        layouts = []
        for file in self.layouts_dir.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    layouts.append({
                        'filename': file.name,
                        'graph_id': data.get('graph_id'),
                        'node_count': data.get('node_count'),
                        'edge_count': data.get('edge_count'),
                        'timestamp': data.get('timestamp'),
                        'algorithm': data.get('metadata', {}).get('algorithm')
                    })
            except: continue
        return layouts
    
    def clear_layout_cache(self, graph_id: str = None):
        if graph_id:
            for file in self.layouts_dir.glob(f"{graph_id}_*.json"): file.unlink()
        else:
            for file in self.layouts_dir.glob("*.json"): file.unlink()


network_service = NetworkService()

@app.get("/")
async def root(): return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/config")
async def get_config():
    return {
        "sql_files": network_service.available_sql_files,
        "metric_modes": {
            "presets": {k: list(v) for k, v in METRIC_PRESETS.items()},
            "categories": {k: v for k, v in METRIC_CATEGORIES.items()}
        },
        "cytoscape_desktop_available": network_service.cytoscape_available,
        "cached_layouts": network_service.list_cached_layouts()
    }

@app.post("/api/load")
def load_network(config: LoadConfig):
    try:
        state = network_service.load_network(config)
        return state
    except Exception as e:
        print(f"Error loading network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/metrics")
def update_metrics(config: MetricsConfig):
    try:
        result = network_service.update_metrics(config)
        return result
    except Exception as e:
        print(f"Error updating metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graphs/{graph_id}/elements")
def get_graph_elements(
    graph_id: str,
    mode: str = Query("full", regex="^(full|nodes_only)$"),
):
    """Return graph elements for Cytoscape.js.

    The `mode` parameter allows loading only nodes for large graphs to
    keep the initial payload light.
    """
    try:
        elements = network_service.get_graph_elements(graph_id, mode=mode)
        return {"elements": elements, "count": len(elements)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graphs/{graph_id}/edges")
def get_graph_edges(
    graph_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(50000, ge=1, le=200000),
):
    """Return a chunk of edges for the given graph.

    This is used to incrementally stream edges to the frontend so that
    the initial graph preview can display nodes quickly while edges
    are loaded in batches.
    """
    try:
        G = network_service.graphs.get(graph_id)
        if G is None:
            raise HTTPException(status_code=404, detail=f"Graph {graph_id} not loaded")

        edges = list(G.edges())
        total = len(edges)
        chunk = edges[offset:offset + limit]

        elements = [
            {
                "group": "edges",
                "data": {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target,
                },
            }
            for (source, target) in chunk
        ]

        return {
            "elements": elements,
            "offset": offset,
            "limit": limit,
            "total": total,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cached-layouts")
async def list_cached_layouts(): return network_service.list_cached_layouts()

@app.delete("/api/cached-layouts")
async def clear_cached_layouts(graph_id: Optional[str] = None):
    network_service.clear_layout_cache(graph_id)
    return {"status": "cleared", "graph_id": graph_id}

@app.get("/api/state")
async def get_current_state():
    if not network_service.graphs: return {"loaded": False}
    total_nodes = sum(len(df) for df in network_service.metrics_dfs.values())
    return {
        "loaded": True,
        "graphs": list(network_service.graphs.keys()),
        "cytoscape_available": network_service.cytoscape_available,
        "node_count": total_nodes
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)