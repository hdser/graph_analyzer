"""
Web viewer with Cytoscape Desktop layout computation and caching
Supports Server-Side Incremental Layout Updates
FIX: Changed async routes to sync (def) to prevent blocking the event loop.
FIX: Metrics caching is now version-aware (v1, v2) to prevent cross-contamination.
FIX: Incremental layout now uses pure Python centroid + spring simulation for speed and consistency.
NEW: Anomaly detection with multiple algorithms
NEW: Composite metrics creation and persistence
NEW: Auto-reload with SSE-based notifications
"""

import os
import json
import time
import hashlib
import random
import re
import math
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from dotenv import load_dotenv

# SSE support
try:
    from sse_starlette.sse import EventSourceResponse
    HAS_SSE = True
except ImportError:
    HAS_SSE = False
    print("[WARNING] sse-starlette not installed. Auto-reload SSE will be disabled.")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from graph_metrics import GraphMetrics, METRIC_CATEGORIES, METRIC_PRESETS

# Import anomaly engine
try:
    from anomaly_engine import AnomalyEngine, CompositeMetricEngine
    HAS_ANOMALY = True
except ImportError:
    HAS_ANOMALY = False
    print("[WARNING] anomaly_engine not found. Anomaly detection will be disabled.")

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
if HAS_ANOMALY:
    print("Anomaly detection engine: AVAILABLE")
if HAS_SSE:
    print("SSE auto-reload: AVAILABLE")
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
# Anomaly Detection Models
# ============================================================================

class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection request."""
    name: str = Field(..., description="Name for the resulting anomaly score metric")
    metrics: List[str] = Field(..., min_length=1, description="Metrics to analyze")
    algorithm: str = Field(..., description="Algorithm name")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm parameters")
    apply_to_graph: bool = Field(default=True, description="Add as node attribute")
    version: Optional[str] = Field(default=None, description="Graph version filter")


class AnomalyDetectionResult(BaseModel):
    """Result of anomaly detection."""
    metric_name: str
    algorithm: str
    n_anomalies: int
    n_total: int
    anomaly_percentage: float
    computation_time: float
    top_anomalies: List[Dict[str, Any]]
    score_statistics: Dict[str, float]
    metrics_used: List[str]
    parameters_used: Dict[str, Any]
    node_updates: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# Composite Metrics Models
# ============================================================================

class CompositeMetricConfig(BaseModel):
    """Configuration for composite metric creation."""
    name: str = Field(..., description="Name for new metric")
    metrics: List[str] = Field(..., min_length=2, max_length=2, description="Source metrics")
    operation: str = Field(..., description="Operation name")
    weights: Optional[List[float]] = Field(default=None, description="For weighted operations")
    normalize: bool = Field(default=False, description="Normalize inputs first")
    save: bool = Field(default=True, description="Persist to cache")
    version: Optional[str] = Field(default=None, description="Graph version")


class CompositeMetricResult(BaseModel):
    """Result of composite metric creation."""
    metric_name: str
    formula: str
    node_updates: List[Dict[str, Any]]
    statistics: Dict[str, float]
    saved: bool
    composite_id: Optional[str] = None


# ============================================================================
# Auto-Reload Models
# ============================================================================

class AutoReloadConfig(BaseModel):
    """Configuration for auto-reload."""
    enabled: bool
    interval_seconds: int = Field(default=300, ge=60, le=3600)
    sql_files: Optional[List[str]] = None
    compute_metrics: bool = Field(default=False)
    metrics_mode: str = Field(default="topology")


class AutoReloadStatus(BaseModel):
    """Current status of auto-reload system."""
    enabled: bool
    interval_seconds: int
    last_reload_time: Optional[str] = None
    next_reload_time: Optional[str] = None
    reload_in_progress: bool
    current_node_count: int
    last_reload_duration: Optional[float] = None
    last_reload_nodes_added: int = 0
    last_reload_nodes_removed: int = 0
    error: Optional[str] = None


# ============================================================================
# Local Spring Layout Algorithm (Pure Python with NumPy)
# ============================================================================

class LocalSpringLayout:
    """
    Fast local spring layout using NumPy for vectorized computation.
    
    This is used for incremental layout updates where we need to place
    new nodes relative to existing anchored nodes. It's much faster than
    calling an external layout service.
    """
    
    def __init__(
        self,
        spring_strength: float = 0.1,
        spring_length: float = 100.0,
        repulsion_strength: float = 5000.0,
        damping: float = 0.8,
        max_velocity: float = 50.0,
        convergence_threshold: float = 0.5,
        max_iterations: int = 100
    ):
        self.spring_strength = spring_strength
        self.spring_length = spring_length
        self.repulsion_strength = repulsion_strength
        self.damping = damping
        self.max_velocity = max_velocity
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
    
    def compute_layout(
        self,
        new_nodes: List[str],
        anchored_positions: Dict[str, Dict[str, float]],
        edges: List[Tuple[str, str]],
        initial_positions: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute positions for new nodes using spring layout.
        
        Args:
            new_nodes: List of node IDs that need positions
            anchored_positions: Dict of node_id -> {x, y} for fixed nodes
            edges: List of (source, target) tuples
            initial_positions: Optional initial positions for new nodes
            
        Returns:
            Dict of node_id -> {x, y} for all nodes (new + anchored)
        """
        if not new_nodes:
            return anchored_positions.copy()
        
        # Combine all nodes
        all_nodes = list(new_nodes) + list(anchored_positions.keys())
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        n_new = len(new_nodes)
        n_total = len(all_nodes)
        
        # Initialize positions array
        positions = np.zeros((n_total, 2), dtype=np.float64)
        
        # Set anchored positions
        for node, pos in anchored_positions.items():
            idx = node_to_idx[node]
            positions[idx] = [pos['x'], pos['y']]
        
        # Set initial positions for new nodes (or use centroid of anchors)
        if initial_positions:
            for node in new_nodes:
                idx = node_to_idx[node]
                if node in initial_positions:
                    positions[idx] = [initial_positions[node]['x'], initial_positions[node]['y']]
                else:
                    # Random position near center of anchors
                    if anchored_positions:
                        cx = np.mean([p['x'] for p in anchored_positions.values()])
                        cy = np.mean([p['y'] for p in anchored_positions.values()])
                        positions[idx] = [cx + random.uniform(-100, 100), cy + random.uniform(-100, 100)]
        else:
            # Use center of anchored nodes with jitter
            if anchored_positions:
                cx = np.mean([p['x'] for p in anchored_positions.values()])
                cy = np.mean([p['y'] for p in anchored_positions.values()])
            else:
                cx, cy = 0.0, 0.0
            
            for i, node in enumerate(new_nodes):
                angle = 2 * math.pi * i / n_new
                r = 100 + random.uniform(-20, 20)
                positions[i] = [cx + r * math.cos(angle), cy + r * math.sin(angle)]
        
        # Build adjacency for spring forces (only edges involving new nodes)
        adjacency = {i: [] for i in range(n_new)}  # Only for new nodes
        for src, tgt in edges:
            if src in node_to_idx and tgt in node_to_idx:
                src_idx = node_to_idx[src]
                tgt_idx = node_to_idx[tgt]
                
                # Only track edges where at least one node is new
                if src_idx < n_new:
                    adjacency[src_idx].append(tgt_idx)
                if tgt_idx < n_new:
                    adjacency[tgt_idx].append(src_idx)
        
        # Velocities for new nodes only
        velocities = np.zeros((n_new, 2), dtype=np.float64)
        
        # Run simulation
        for iteration in range(self.max_iterations):
            forces = np.zeros((n_new, 2), dtype=np.float64)
            
            # Calculate spring forces (attraction along edges)
            for i in range(n_new):
                for j in adjacency[i]:
                    diff = positions[j] - positions[i]
                    dist = np.linalg.norm(diff)
                    if dist > 0.1:  # Avoid division by zero
                        # Spring force: F = k * (distance - rest_length) * direction
                        force_magnitude = self.spring_strength * (dist - self.spring_length)
                        force = force_magnitude * (diff / dist)
                        forces[i] += force
            
            # Calculate repulsion forces between new nodes
            for i in range(n_new):
                for j in range(i + 1, n_new):
                    diff = positions[j] - positions[i]
                    dist_sq = np.sum(diff ** 2)
                    if dist_sq > 0.1:  # Avoid division by zero
                        # Coulomb repulsion: F = k / distance^2
                        force_magnitude = self.repulsion_strength / dist_sq
                        force = force_magnitude * (diff / math.sqrt(dist_sq))
                        forces[i] -= force
                        forces[j] += force
            
            # Update velocities and positions (only for new nodes)
            velocities = (velocities + forces) * self.damping
            
            # Clamp velocity
            velocity_magnitudes = np.linalg.norm(velocities, axis=1, keepdims=True)
            velocity_magnitudes = np.maximum(velocity_magnitudes, 0.001)  # Avoid division by zero
            velocity_scale = np.minimum(1.0, self.max_velocity / velocity_magnitudes)
            velocities *= velocity_scale
            
            # Update positions
            positions[:n_new] += velocities
            
            # Check convergence
            max_displacement = np.max(np.abs(velocities))
            if max_displacement < self.convergence_threshold:
                print(f"[SPRING LAYOUT] Converged after {iteration + 1} iterations")
                break
        
        # Convert back to dict format
        result = {}
        for node in all_nodes:
            idx = node_to_idx[node]
            result[node] = {'x': float(positions[idx, 0]), 'y': float(positions[idx, 1])}
        
        return result


# ============================================================================
# Auto-Reload Manager
# ============================================================================

class AutoReloadManager:
    """
    Manages automatic background reloading of graph data.
    
    Features:
    - Configurable interval (60-3600 seconds)
    - SSE-based event broadcasting
    - Atomic state updates
    - Diff computation (added/removed nodes)
    - Thread-safe operation
    """
    
    def __init__(self, network_service: 'NetworkService'):
        self.network_service = network_service
        self.enabled = False
        self.interval_seconds = 300
        self.sql_files: List[str] = []
        self.compute_metrics = False
        self.metrics_mode = "topology"
        
        # State tracking
        self.last_reload_time: Optional[datetime] = None
        self.next_reload_time: Optional[datetime] = None
        self.reload_in_progress = False
        self.last_error: Optional[str] = None
        self.last_reload_duration: Optional[float] = None
        self.last_nodes_added: int = 0
        self.last_nodes_removed: int = 0
        
        # Threading
        self._task: Optional[asyncio.Task] = None
        self._state_lock = threading.Lock()
        self._stop_event = asyncio.Event()
        
        # SSE subscribers
        self._subscribers: List[asyncio.Queue] = []
    
    async def start(self, config: AutoReloadConfig) -> AutoReloadStatus:
        """Start auto-reload with given configuration."""
        # Stop any existing task
        await self.stop()
        
        self.enabled = config.enabled
        self.interval_seconds = config.interval_seconds
        self.sql_files = config.sql_files or []
        self.compute_metrics = config.compute_metrics
        self.metrics_mode = config.metrics_mode
        
        if not self.enabled:
            return self.get_status()
        
        if not self.sql_files:
            # Use currently loaded graphs
            self.sql_files = [f"{gid}.sql" for gid in self.network_service.graphs.keys()]
        
        if not self.sql_files:
            self.enabled = False
            self.last_error = "No SQL files specified and no graphs loaded"
            return self.get_status()
        
        # Calculate next reload time
        self.next_reload_time = datetime.now() + timedelta(seconds=self.interval_seconds)
        self.last_error = None
        self._stop_event.clear()
        
        # Start background task
        self._task = asyncio.create_task(self._reload_loop())
        
        print(f"[AUTO-RELOAD] Started with interval {self.interval_seconds}s")
        return self.get_status()
    
    async def stop(self) -> AutoReloadStatus:
        """Stop auto-reload."""
        self.enabled = False
        self._stop_event.set()
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self._task = None
        self.next_reload_time = None
        
        print("[AUTO-RELOAD] Stopped")
        return self.get_status()
    
    def get_status(self) -> AutoReloadStatus:
        """Get current status."""
        total_nodes = sum(
            G.number_of_nodes() 
            for G in self.network_service.graphs.values()
        ) if self.network_service.graphs else 0
        
        return AutoReloadStatus(
            enabled=self.enabled,
            interval_seconds=self.interval_seconds,
            last_reload_time=self.last_reload_time.isoformat() if self.last_reload_time else None,
            next_reload_time=self.next_reload_time.isoformat() if self.next_reload_time else None,
            reload_in_progress=self.reload_in_progress,
            current_node_count=total_nodes,
            last_reload_duration=self.last_reload_duration,
            last_reload_nodes_added=self.last_nodes_added,
            last_reload_nodes_removed=self.last_nodes_removed,
            error=self.last_error
        )
    
    def subscribe(self) -> asyncio.Queue:
        """Subscribe to reload events. Returns queue for SSE."""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return queue
    
    def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from reload events."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)
    
    async def _broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to all subscribers."""
        event = {"type": event_type, "data": data}
        for queue in self._subscribers:
            try:
                await queue.put(event)
            except Exception as e:
                print(f"[AUTO-RELOAD] Error broadcasting to subscriber: {e}")
    
    async def _reload_loop(self):
        """Main reload loop - runs in background."""
        while self.enabled and not self._stop_event.is_set():
            try:
                # Wait for interval
                sleep_time = self.interval_seconds
                if self.next_reload_time:
                    remaining = (self.next_reload_time - datetime.now()).total_seconds()
                    sleep_time = max(1, remaining)
                
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=sleep_time
                    )
                    # Stop event was set
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, proceed with reload
                    pass
                
                if not self.enabled:
                    break
                
                # Perform reload
                result = await self._perform_reload()
                
                if result:
                    self.last_reload_time = datetime.now()
                    self.next_reload_time = datetime.now() + timedelta(seconds=self.interval_seconds)
                    self.last_reload_duration = result.get('duration_seconds', 0)
                    self.last_nodes_added = len(result.get('nodes_added', []))
                    self.last_nodes_removed = len(result.get('nodes_removed', []))
                    self.last_error = None
                    
                    # Broadcast completion
                    await self._broadcast_event('reload_complete', result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error = str(e)
                print(f"[AUTO-RELOAD] Error in reload loop: {e}")
                import traceback
                traceback.print_exc()
                
                await self._broadcast_event('reload_error', {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Wait before retrying
                await asyncio.sleep(30)
    
    async def _perform_reload(self) -> Optional[Dict[str, Any]]:
        """
        Perform a single reload cycle.
        
        Returns dict with:
        - nodes_added: List[str]
        - nodes_removed: List[str]
        - total_nodes: int
        - duration_seconds: float
        - graphs_updated: List[str]
        """
        if self.reload_in_progress:
            return None
        
        self.reload_in_progress = True
        start_time = time.time()
        
        try:
            # Broadcast start
            await self._broadcast_event('reload_started', {
                'timestamp': datetime.now().isoformat(),
                'graphs': list(self.network_service.graphs.keys())
            })
            
            # Get current node sets for diff
            old_nodes: Dict[str, Set[str]] = {}
            for gid, G in self.network_service.graphs.items():
                old_nodes[gid] = set(G.nodes())
            
            # Load fresh data
            config = LoadConfig(
                sql_files=self.sql_files,
                use_cached_layout=True,
                skip_sql=False
            )
            
            # Run in executor to not block event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.network_service.load_network(config)
            )
            
            # Compute diff
            all_added = []
            all_removed = []
            
            for gid, G in self.network_service.graphs.items():
                new_node_set = set(G.nodes())
                old_node_set = old_nodes.get(gid, set())
                
                added = list(new_node_set - old_node_set)
                removed = list(old_node_set - new_node_set)
                
                all_added.extend(added)
                all_removed.extend(removed)
            
            # Optionally compute metrics
            if self.compute_metrics and all_added:
                metrics_config = MetricsConfig(
                    metrics_mode=self.metrics_mode,
                    metrics_graph_id=None
                )
                await loop.run_in_executor(
                    None,
                    lambda: self.network_service.update_metrics(metrics_config)
                )
            
            duration = time.time() - start_time
            total_nodes = sum(G.number_of_nodes() for G in self.network_service.graphs.values())
            
            result = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'nodes_added': all_added,
                'nodes_removed': all_removed,
                'total_nodes': total_nodes,
                'graphs_updated': list(self.network_service.graphs.keys())
            }
            
            print(f"[AUTO-RELOAD] Completed in {duration:.2f}s. "
                  f"+{len(all_added)} nodes, -{len(all_removed)} nodes")
            
            return result
            
        except Exception as e:
            print(f"[AUTO-RELOAD] Reload failed: {e}")
            raise
        finally:
            self.reload_in_progress = False


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
        
        # Initialize local spring layout for incremental updates
        self.local_spring_layout = LocalSpringLayout(
            spring_strength=0.08,
            spring_length=80.0,
            repulsion_strength=3000.0,
            damping=0.85,
            max_velocity=40.0,
            convergence_threshold=0.3,
            max_iterations=80
        )
        
        # Initialize anomaly and composite engines
        if HAS_ANOMALY:
            self.anomaly_engine = AnomalyEngine()
            self.composite_engine = CompositeMetricEngine(
                cache_path=str(self.cache_dir / "composite_metrics.json")
            )
        else:
            self.anomaly_engine = None
            self.composite_engine = None
        
        # Initialize auto-reload manager
        self.auto_reload_manager = AutoReloadManager(self)
    
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
    
    def _calculate_centroid_positions(
        self, 
        new_nodes: List[str], 
        anchors: Set[str], 
        current_layout: Dict[str, Dict[str, float]], 
        G: nx.DiGraph
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate initial positions for new nodes based on the centroid of their neighbors.
        
        If a new node has neighbors in the current layout, place it at their centroid
        with some jitter. If it has no positioned neighbors, place it near the graph center.
        """
        initial_positions = {}
        
        # Calculate the center of the current layout for fallback
        if current_layout:
            center_x = np.mean([p['x'] for p in current_layout.values()])
            center_y = np.mean([p['y'] for p in current_layout.values()])
        else:
            center_x, center_y = 0.0, 0.0
        
        for node in new_nodes:
            neighbors_positions = []
            
            if G.has_node(node):
                # Get positions of outgoing neighbors
                for n in G.successors(node):
                    if n in current_layout:
                        neighbors_positions.append(current_layout[n])
                
                # Get positions of incoming neighbors
                for n in G.predecessors(node):
                    if n in current_layout:
                        neighbors_positions.append(current_layout[n])
            
            if neighbors_positions:
                # Place at centroid of neighbors with jitter
                avg_x = sum(p['x'] for p in neighbors_positions) / len(neighbors_positions)
                avg_y = sum(p['y'] for p in neighbors_positions) / len(neighbors_positions)
                
                # Add jitter proportional to number of neighbors (more neighbors = less jitter)
                jitter_scale = 50.0 / math.sqrt(len(neighbors_positions))
                initial_positions[node] = {
                    'x': avg_x + random.uniform(-jitter_scale, jitter_scale),
                    'y': avg_y + random.uniform(-jitter_scale, jitter_scale)
                }
            else:
                # No positioned neighbors - place near graph center with larger jitter
                initial_positions[node] = {
                    'x': center_x + random.uniform(-200, 200), 
                    'y': center_y + random.uniform(-200, 200)
                }
        
        return initial_positions

    def compute_incremental_layout(
        self, 
        graph_id: str, 
        new_nodes: List[str], 
        current_layout: Dict[str, Dict[str, float]], 
        G: nx.DiGraph
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute positions for new nodes using local spring layout.
        
        This uses a pure Python implementation that:
        1. Places new nodes at the centroid of their neighbors
        2. Runs a local spring simulation to resolve overlaps
        3. Keeps existing nodes fixed (anchored)
        
        This is much faster and more consistent than calling an external layout service.
        """
        start_time = time.time()
        
        if not new_nodes:
            return {}
        
        new_nodes_set = set(new_nodes)
        anchors = set()
        edges_list: List[Tuple[str, str]] = []
        
        # Collect edges involving new nodes and identify anchor nodes
        for node in new_nodes:
            if G.has_node(node):
                # Outgoing edges
                for _, target in G.out_edges(node):
                    edges_list.append((node, target))
                    if target not in new_nodes_set and target in current_layout:
                        anchors.add(target)
                
                # Incoming edges
                for source, _ in G.in_edges(node):
                    edges_list.append((source, node))
                    if source not in new_nodes_set and source in current_layout:
                        anchors.add(source)
        
        # Also add edges between new nodes
        for node in new_nodes:
            if G.has_node(node):
                for _, target in G.out_edges(node):
                    if target in new_nodes_set:
                        edges_list.append((node, target))
        
        # Remove duplicate edges
        edges_list = list(set(edges_list))
        
        print(f"[INCREMENTAL LAYOUT] Processing {len(new_nodes)} new nodes with {len(anchors)} anchors")
        print(f"[INCREMENTAL LAYOUT] Found {len(edges_list)} relevant edges")
        
        # Get anchor positions
        anchored_positions = {n: current_layout[n] for n in anchors if n in current_layout}
        
        # Calculate initial positions using centroid method
        initial_positions = self._calculate_centroid_positions(new_nodes, anchors, current_layout, G)
        
        # Run local spring layout
        new_positions = self.local_spring_layout.compute_layout(
            new_nodes=new_nodes,
            anchored_positions=anchored_positions,
            edges=edges_list,
            initial_positions=initial_positions
        )
        
        # Extract only the new node positions (not anchors)
        result = {node: new_positions[node] for node in new_nodes if node in new_positions}
        
        elapsed = time.time() - start_time
        print(f"[INCREMENTAL LAYOUT] Completed in {elapsed:.3f}s")
        
        return result

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

        # Incremental updates pre-calc using LOCAL SPRING LAYOUT
        for graph_id in new_graphs:
            G = new_graphs[graph_id]
            layout = new_layouts.get(graph_id, {})
            existing_nodes_in_layout = set(layout.keys())
            all_nodes_in_graph = set(G.nodes())
            missing_nodes = list(all_nodes_in_graph - existing_nodes_in_layout)
            
            if missing_nodes:
                print(f"[PRE-COMPUTE] Graph {graph_id}: Finding positions for {len(missing_nodes)} missing nodes using local spring layout...")
                try:
                    # Use our new local spring layout instead of calling the service
                    new_positions = self.compute_incremental_layout(graph_id, missing_nodes, layout, G)
                    layout.update(new_positions)
                    new_layouts[graph_id] = layout
                    
                    # Save updated cache
                    self.save_layout_cache(
                        graph_id, G.number_of_nodes(), G.number_of_edges(), layout, 
                        {'updated': True, 'update_time': datetime.now().isoformat(), 'algorithm': 'local-spring'}
                    )
                except Exception as e:
                    print(f"[PRE-COMPUTE] Error: {e}")
                    import traceback
                    traceback.print_exc()

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

    def get_metrics_dataframe(self, version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get the metrics DataFrame for a specific version or the first available."""
        if version and version in self.metrics_dfs:
            return self.metrics_dfs[version]
        elif self.metrics_dfs:
            return list(self.metrics_dfs.values())[0]
        return None
    
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

# ============================================================================
# Core Endpoints
# ============================================================================

@app.get("/")
async def root(): 
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/config")
async def get_config():
    config = {
        "sql_files": network_service.available_sql_files,
        "metric_modes": {
            "presets": {k: list(v) for k, v in METRIC_PRESETS.items()},
            "categories": {k: v for k, v in METRIC_CATEGORIES.items()}
        },
        "cytoscape_desktop_available": network_service.cytoscape_available,
        "cached_layouts": network_service.list_cached_layouts(),
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE
    }
    
    # Add anomaly algorithms if available
    if HAS_ANOMALY and network_service.anomaly_engine:
        config["anomaly_algorithms"] = AnomalyEngine.get_available_algorithms()
        config["composite_operations"] = CompositeMetricEngine.get_available_operations()
    
    return config

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
async def list_cached_layouts(): 
    return network_service.list_cached_layouts()

@app.delete("/api/cached-layouts")
async def clear_cached_layouts(graph_id: Optional[str] = None):
    network_service.clear_layout_cache(graph_id)
    return {"status": "cleared", "graph_id": graph_id}

@app.get("/api/state")
async def get_current_state():
    if not network_service.graphs: 
        return {"loaded": False}
    total_nodes = sum(len(df) for df in network_service.metrics_dfs.values())
    return {
        "loaded": True,
        "graphs": list(network_service.graphs.keys()),
        "cytoscape_available": network_service.cytoscape_available,
        "node_count": total_nodes,
        "anomaly_available": HAS_ANOMALY,
        "auto_reload_available": HAS_SSE
    }


# ============================================================================
# Anomaly Detection Endpoints
# ============================================================================

@app.get("/api/anomaly/algorithms")
async def get_anomaly_algorithms():
    """Get available anomaly detection algorithms with parameters."""
    if not HAS_ANOMALY:
        raise HTTPException(status_code=503, detail="Anomaly detection not available. Install scikit-learn.")
    return AnomalyEngine.get_available_algorithms()


@app.post("/api/anomaly/detect")
def detect_anomalies(config: AnomalyDetectionConfig):
    """
    Run anomaly detection on graph metrics.
    
    Returns anomaly scores as new metric that can be used for
    coloring/filtering in the visualization.
    """
    if not HAS_ANOMALY or not network_service.anomaly_engine:
        raise HTTPException(status_code=503, detail="Anomaly detection not available")
    
    if not network_service.graphs:
        raise HTTPException(status_code=400, detail="No graphs loaded. Please load networks first.")
    
    try:
        # Get metrics DataFrame
        version = config.version
        if not version:
            # Use first available version
            version = list(network_service.metrics_dfs.keys())[0] if network_service.metrics_dfs else None
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No metrics data available. Run metrics first.")
        
        # Run anomaly detection
        result = network_service.anomaly_engine.detect_anomalies(
            df=df,
            metrics=config.metrics,
            algorithm=config.algorithm,
            parameters=config.parameters
        )
        
        # Prepare response
        response = AnomalyDetectionResult(
            metric_name=config.name,
            algorithm=result.algorithm,
            n_anomalies=result.n_anomalies,
            n_total=result.n_total,
            anomaly_percentage=(result.n_anomalies / result.n_total * 100) if result.n_total > 0 else 0,
            computation_time=result.computation_time,
            top_anomalies=result.top_anomalies,
            score_statistics=result.statistics,
            metrics_used=result.metrics_used,
            parameters_used=result.parameters
        )
        
        # Apply to graph if requested
        if config.apply_to_graph:
            node_updates = []
            
            for gid, G in network_service.graphs.items():
                graph_version = network_service._extract_version(gid)
                if version and graph_version != version:
                    continue
                
                for node_id, score in result.scores.items():
                    if G.has_node(node_id):
                        G.nodes[node_id][config.name] = score
                        G.nodes[node_id][f"{config.name}_is_anomaly"] = result.binary_labels.get(node_id, False)
                        node_updates.append({
                            'id': node_id,
                            config.name: score,
                            f"{config.name}_is_anomaly": result.binary_labels.get(node_id, False)
                        })
            
            response.node_updates = node_updates
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Anomaly detection error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Composite Metrics Endpoints
# ============================================================================

@app.get("/api/anomaly/composites")
async def get_composite_metrics(version: Optional[str] = None):
    """Get list of saved composite metrics."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(status_code=503, detail="Composite metrics not available")
    
    return network_service.composite_engine.get_saved_composites(version)


@app.get("/api/metrics/composite/operations")
async def get_composite_operations():
    """Get available composite metric operations."""
    if not HAS_ANOMALY:
        raise HTTPException(status_code=503, detail="Composite metrics not available")
    return CompositeMetricEngine.get_available_operations()


@app.post("/api/metrics/composite")
def create_composite_metric(config: CompositeMetricConfig):
    """
    Create a composite metric from existing metrics.
    
    Optionally saves to cache for reuse across sessions.
    """
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(status_code=503, detail="Composite metrics not available")
    
    if not network_service.graphs:
        raise HTTPException(status_code=400, detail="No graphs loaded. Please load networks first.")
    
    try:
        # Get metrics DataFrame
        version = config.version
        if not version:
            version = list(network_service.metrics_dfs.keys())[0] if network_service.metrics_dfs else None
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No metrics data available. Run metrics first.")
        
        # Create composite metric
        result_series, metadata = network_service.composite_engine.create_composite(
            df=df,
            name=config.name,
            metrics=config.metrics,
            operation=config.operation,
            weights=config.weights,
            normalize=config.normalize,
            save=config.save,
            version=version or "default"
        )
        
        # Apply to graphs
        node_updates = []
        
        for gid, G in network_service.graphs.items():
            graph_version = network_service._extract_version(gid)
            if version and graph_version != version:
                continue
            
            for node_id, value in result_series.items():
                if G.has_node(node_id):
                    G.nodes[node_id][config.name] = float(value)
                    node_updates.append({
                        'id': node_id,
                        config.name: float(value)
                    })
        
        return CompositeMetricResult(
            metric_name=config.name,
            formula=metadata['formula'],
            node_updates=node_updates,
            statistics=metadata['statistics'],
            saved=metadata.get('saved', False),
            composite_id=metadata.get('id')
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Composite metric error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/metrics/composite/{composite_id}")
async def delete_composite_metric(composite_id: str):
    """Delete a saved composite metric."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(status_code=503, detail="Composite metrics not available")
    
    success = network_service.composite_engine.delete_composite(composite_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Composite {composite_id} not found")
    
    return {"status": "deleted", "composite_id": composite_id}


@app.post("/api/metrics/composite/{composite_id}/apply")
def apply_composite_metric(composite_id: str, version: Optional[str] = None):
    """Apply a saved composite metric to current graph data."""
    if not HAS_ANOMALY or not network_service.composite_engine:
        raise HTTPException(status_code=503, detail="Composite metrics not available")
    
    if not network_service.graphs:
        raise HTTPException(status_code=400, detail="No graphs loaded")
    
    try:
        # Get the composite config
        composite = network_service.composite_engine.get_composite_by_id(composite_id)
        if not composite:
            raise HTTPException(status_code=404, detail=f"Composite {composite_id} not found")
        
        # Get metrics DataFrame
        if not version:
            version = composite.get('version', 'default')
        
        df = network_service.get_metrics_dataframe(version)
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No metrics data available")
        
        # Apply composite
        result_series, metadata = network_service.composite_engine.apply_saved_composite(
            composite_id, df
        )
        
        if result_series is None:
            raise HTTPException(status_code=404, detail=f"Composite {composite_id} not found")
        
        # Apply to graphs
        node_updates = []
        metric_name = composite['name']
        
        for gid, G in network_service.graphs.items():
            graph_version = network_service._extract_version(gid)
            if version != 'default' and graph_version != version:
                continue
            
            for node_id, value in result_series.items():
                if G.has_node(node_id):
                    G.nodes[node_id][metric_name] = float(value)
                    node_updates.append({
                        'id': node_id,
                        metric_name: float(value)
                    })
        
        return {
            "metric_name": metric_name,
            "node_updates": node_updates,
            "count": len(node_updates)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Auto-Reload Endpoints
# ============================================================================

@app.post("/api/auto-reload/start")
async def start_auto_reload(config: AutoReloadConfig):
    """Start automatic background reloading."""
    if not HAS_SSE:
        raise HTTPException(status_code=503, detail="SSE not available. Install sse-starlette.")
    
    try:
        status = await network_service.auto_reload_manager.start(config)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto-reload/stop")
async def stop_auto_reload():
    """Stop automatic background reloading."""
    try:
        status = await network_service.auto_reload_manager.stop()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auto-reload/status")
async def get_auto_reload_status():
    """Get current auto-reload status."""
    return network_service.auto_reload_manager.get_status()


@app.get("/api/auto-reload/events")
async def auto_reload_events(request: Request):
    """
    SSE endpoint for real-time reload notifications.
    
    Event types:
    - reload_started
    - reload_progress
    - reload_complete
    - reload_error
    - status_update
    """
    if not HAS_SSE:
        raise HTTPException(status_code=503, detail="SSE not available")
    
    async def event_generator():
        queue = network_service.auto_reload_manager.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event["type"],
                        "data": json.dumps(event["data"])
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": "{}"}
        finally:
            network_service.auto_reload_manager.unsubscribe(queue)
    
    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)