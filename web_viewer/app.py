"""
Web viewer with Cytoscape Desktop layout computation and caching
Supports Server-Side Incremental Layout Updates
"""

import os
import json
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException
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

class GraphConfig(BaseModel):
    sql_files: List[str]
    metrics_mode: str = "basic"
    metrics_graph_id: Optional[str] = None
    layout_algorithm: str = "auto"  # "auto", "cytoscape-desktop", "fcose", etc.
    use_cached_layout: bool = True  # Try to use cached layout first


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
        
        # Create directories
        self.cache_dir.mkdir(exist_ok=True)
        self.layouts_dir.mkdir(exist_ok=True)
        
        self.edge_layers = {}
        self.metrics_df = None
        self.layouts = {}
        self.graphs = {}
        self.current_config = None
        
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
            raise ValueError("Missing DB credentials")
        
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
            p4c.layout_network("force-directed", network=net_suid)
            
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
        
        # --- FIXED: Robust handling of input type to prevent list index errors ---
        if isinstance(df_edges, pd.DataFrame):
            # Standard full layout case
            edges = [{"source": row['source'], "target": row['target']} for _, row in df_edges.iterrows()]
            node_ids = list(set(df_edges['source'].tolist() + df_edges['target'].tolist()))
        elif isinstance(df_edges, list):
            # Incremental layout case (list of dicts)
            edges = df_edges
            # Extract unique nodes from the list of dicts
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
                       df_metrics_all: pd.DataFrame, config: GraphConfig) -> Dict:
        node_count = len(df_metrics_all)
        edge_count = len(df_edges)
        
        if config.use_cached_layout:
            cached_layout = self.get_cached_layout(graph_id, node_count, edge_count)
            if cached_layout: return cached_layout
        
        if config.layout_algorithm == "auto":
            if self.cytoscape_available and edge_count < 500000:
                try:
                    return self.compute_layout_via_cytoscape_desktop(graph_id, df_edges, df_metrics_all)
                except Exception as e:
                    print(f"[LAYOUT] Cytoscape Desktop failed: {e}")
            return self.compute_layout_via_service(graph_id, df_edges, "fcose")
        elif config.layout_algorithm == "cytoscape-desktop":
            if not self.cytoscape_available:
                return self.compute_layout_via_service(graph_id, df_edges, "fcose")
            return self.compute_layout_via_cytoscape_desktop(graph_id, df_edges, df_metrics_all)
        else:
            return self.compute_layout_via_service(graph_id, df_edges, config.layout_algorithm)
    
    def load_edge_layers_from_sql(self, sql_files: List[str]) -> dict:
        edge_layers = {}
        for filename in sql_files:
            sql_path = self.sql_dir / filename
            if not sql_path.exists(): continue
            graph_id = sql_path.stem
            with open(sql_path, 'r') as f: query = f.read()
            df = pd.read_sql_query(query, self.db_engine)
            edge_layers[graph_id] = df
        return edge_layers
    
    def compute_metrics_for_shared_avatars(self, edge_layers: dict,
                                          metrics_graph_id: str,
                                          metrics_mode: str = None) -> pd.DataFrame:
        if metrics_graph_id not in edge_layers:
            raise ValueError(f"metrics_graph_id='{metrics_graph_id}' not found")
        
        df_metrics_edges = edge_layers[metrics_graph_id]
        G = nx.DiGraph()
        G.add_edges_from(df_metrics_edges[["source", "target"]].itertuples(index=False, name=None))
        
        all_avatars = set()
        for gid, df_edges in edge_layers.items():
            all_avatars.update(df_edges["source"].unique())
            all_avatars.update(df_edges["target"].unique())
        
        if G.number_of_nodes() > 50000:
            if not (',' in (metrics_mode or '') or metrics_mode in ["basic", "topology", "essential"]):
                metrics_mode = "basic"
        
        metrics_calc = GraphMetrics(G, n_jobs=4, metrics_mode=metrics_mode or "all")
        df_metrics = metrics_calc.compute_all()
        
        df_all = pd.DataFrame({"avatar": list(all_avatars)})
        df_metrics_all = df_all.merge(df_metrics, on="avatar", how="left")
        metric_cols = [c for c in df_metrics_all.columns if c != "avatar"]
        df_metrics_all[metric_cols] = df_metrics_all[metric_cols].replace([np.inf, -np.inf], 0).fillna(0)
        return df_metrics_all
    
    async def load_network(self, config: GraphConfig) -> NetworkState:
        """
        Load network with background computation and atomic state swap.
        Ensures the current graph remains valid until the new one is fully ready.
        """
        start_time = time.time()
        
        # --- PHASE 1: Compute everything in local variables (Background) ---
        
        new_edge_layers = self.load_edge_layers_from_sql(config.sql_files)
        
        metrics_graph_id = config.metrics_graph_id
        if not metrics_graph_id and new_edge_layers:
            metrics_graph_id = list(new_edge_layers.keys())[0]
        
        new_metrics_df = self.compute_metrics_for_shared_avatars(
            edge_layers=new_edge_layers,
            metrics_graph_id=metrics_graph_id,
            metrics_mode=config.metrics_mode
        )
        
        layout_start = time.time()
        new_layouts = {}
        new_graphs = {}
        layout_cached = False
        
        # Build graphs and layouts locally
        for graph_id, df_edges in new_edge_layers.items():
            G = nx.DiGraph()
            metrics_dict = new_metrics_df.set_index('avatar').to_dict('index')
            for avatar, attrs in metrics_dict.items():
                clean_attrs = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in attrs.items()}
                G.add_node(avatar, **clean_attrs)
            
            for _, row in df_edges.iterrows():
                G.add_edge(row['source'], row['target'])
            
            new_graphs[graph_id] = G
            
            positions = self.compute_layout(graph_id, df_edges, new_metrics_df, config)
            new_layouts[graph_id] = positions
            
            if config.use_cached_layout:
                cache_key = self.get_layout_cache_key(graph_id, G.number_of_nodes(), G.number_of_edges())
                if (self.layouts_dir / f"{cache_key}.json").exists():
                    layout_cached = True

        # Pre-compute incremental updates locally
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
                    
                    # Update cache
                    self.save_layout_cache(
                        graph_id, G.number_of_nodes(), G.number_of_edges(), layout, 
                        {'updated': True, 'update_time': datetime.now().isoformat()}
                    )
                except Exception as e:
                    print(f"[PRE-COMPUTE] Error: {e}")

        layout_time = time.time() - layout_start
        total_time = time.time() - start_time

        # --- PHASE 2: Atomic State Swap ---
        self.edge_layers = new_edge_layers
        self.metrics_df = new_metrics_df
        self.graphs = new_graphs
        self.layouts = new_layouts
        self.current_config = config
        
        return NetworkState(
            loaded_graphs=list(self.graphs.keys()),
            current_graph=list(self.graphs.keys())[0] if self.graphs else None,
            node_count=len(self.metrics_df) if self.metrics_df is not None else 0,
            edge_count=sum(len(df) for df in self.edge_layers.values()),
            metrics_computed=list(self.metrics_df.columns) if self.metrics_df is not None else [],
            computation_time=total_time,
            layout_computation_time=layout_time,
            layout_algorithm=config.layout_algorithm,
            layout_cached=layout_cached
        )
    
    def get_graph_elements(self, graph_id: str):
        if graph_id not in self.graphs:
            raise ValueError(f"Graph {graph_id} not loaded")
        
        G = self.graphs[graph_id]
        layout = self.layouts.get(graph_id, {})
        elements = []
        
        for node in G.nodes():
            node_data = dict(G.nodes[node])
            clean_data = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v) for k, v in node_data.items()}
            clean_data['id'] = node
            clean_data['label'] = node[:10] + "..." if len(node) > 12 else node
            
            node_element = {"group": "nodes", "data": clean_data}
            if node in layout:
                node_element["position"] = layout[node]
            elements.append(node_element)
        
        for source, target in G.edges():
            edge_element = {
                "group": "edges",
                "data": {
                    "id": f"{source}->{target}",
                    "source": source,
                    "target": target
                }
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
async def load_network(config: GraphConfig):
    try:
        state = await network_service.load_network(config)
        return state
    except Exception as e:
        print(f"Error loading network: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graphs/{graph_id}/elements")
async def get_graph_elements(graph_id: str):
    try:
        elements = network_service.get_graph_elements(graph_id)
        return {"elements": elements, "count": len(elements)}
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
    return {
        "loaded": True,
        "graphs": list(network_service.graphs.keys()),
        "cytoscape_available": network_service.cytoscape_available,
        "node_count": len(network_service.metrics_df) if network_service.metrics_df is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)