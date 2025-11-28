"""
Cache Service

Handles caching for layouts, data, and metrics.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from backend.config import settings


class CacheService:
    """
    Service for managing layout, data, and metrics caches.
    
    Cache structure:
    - cache/layouts/{graph_id}_{nodes}n_{edges}e.json
    - cache/data/{graph_id}_edges.csv
    - cache/data/node_metrics_{version}.csv
    """
    
    def __init__(self):
        """Initialize cache service with configured paths."""
        self.layouts_dir = settings.LAYOUTS_DIR
        self.data_dir = settings.DATA_CACHE_DIR
        
        # Ensure directories exist
        self.layouts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Layout Cache
    # =========================================================================
    
    def get_layout_cache_key(self, graph_id: str, node_count: int, edge_count: int) -> str:
        """Generate a cache key for a layout."""
        return f"{graph_id}_{node_count}n_{edge_count}e"
    
    def get_cached_layout(
        self, 
        graph_id: str, 
        node_count: int, 
        edge_count: int,
        tolerance: float = 0.1
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get cached layout if available.
        
        Args:
            graph_id: Graph identifier
            node_count: Current node count
            edge_count: Current edge count
            tolerance: Allowed size difference ratio (default 10%)
            
        Returns:
            Layout dictionary {node_id: {x, y}} or None if not cached
        """
        # Try exact match first
        cache_key = self.get_layout_cache_key(graph_id, node_count, edge_count)
        cache_file = self.layouts_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                print(f"[CACHE] Exact layout match: {cache_file.name}")
                return data.get('positions', data)
            except Exception as e:
                print(f"[CACHE] Error loading layout: {e}")
        
        # Try similar size match
        for cache_file in self.layouts_dir.glob(f"{graph_id}_*.json"):
            try:
                # Parse node/edge counts from filename
                parts = cache_file.stem.split('_')
                if len(parts) >= 3:
                    cached_nodes = int(parts[-2].replace('n', ''))
                    cached_edges = int(parts[-1].replace('e', ''))
                    
                    # Check if within tolerance
                    node_ratio = abs(cached_nodes - node_count) / max(node_count, 1)
                    edge_ratio = abs(cached_edges - edge_count) / max(edge_count, 1)
                    
                    if node_ratio <= tolerance and edge_ratio <= tolerance:
                        with open(cache_file, 'r') as f:
                            data = json.load(f)
                        print(f"[CACHE] Similar layout match: {cache_file.name} "
                              f"(nodes: {cached_nodes} vs {node_count}, edges: {cached_edges} vs {edge_count})")
                        return data.get('positions', data)
            except Exception:
                continue
        
        return None
    
    def save_layout_cache(
        self, 
        graph_id: str, 
        node_count: int, 
        edge_count: int,
        positions: Dict[str, Dict[str, float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save layout to cache.
        
        Args:
            graph_id: Graph identifier
            node_count: Number of nodes
            edge_count: Number of edges
            positions: Layout positions {node_id: {x, y}}
            metadata: Optional metadata to store
            
        Returns:
            Cache file path
        """
        cache_key = self.get_layout_cache_key(graph_id, node_count, edge_count)
        cache_file = self.layouts_dir / f"{cache_key}.json"
        
        data = {
            'positions': positions,
            'metadata': metadata or {},
            'node_count': node_count,
            'edge_count': edge_count,
            'graph_id': graph_id
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        
        print(f"[CACHE] Saved layout: {cache_file.name}")
        return str(cache_file)
    
    def list_cached_layouts(self) -> List[Dict[str, Any]]:
        """List all cached layouts with metadata."""
        layouts = []
        for cache_file in self.layouts_dir.glob("*.json"):
            try:
                parts = cache_file.stem.split('_')
                if len(parts) >= 3:
                    graph_id = '_'.join(parts[:-2])
                    nodes = int(parts[-2].replace('n', ''))
                    edges = int(parts[-1].replace('e', ''))
                    
                    layouts.append({
                        'filename': cache_file.name,
                        'graph_id': graph_id,
                        'node_count': nodes,
                        'edge_count': edges,
                        'size_mb': cache_file.stat().st_size / (1024 * 1024)
                    })
            except Exception:
                continue
        
        return sorted(layouts, key=lambda x: x.get('graph_id', ''))
    
    def clear_layout_cache(self, graph_id: Optional[str] = None):
        """
        Clear layout cache.
        
        Args:
            graph_id: Clear only for this graph (None = clear all)
        """
        if graph_id:
            pattern = f"{graph_id}_*.json"
        else:
            pattern = "*.json"
        
        for cache_file in self.layouts_dir.glob(pattern):
            cache_file.unlink()
            print(f"[CACHE] Deleted: {cache_file.name}")
    
    # =========================================================================
    # Data Cache
    # =========================================================================
    
    def save_data_cache(
        self, 
        graph_id: str, 
        edges_df: pd.DataFrame
    ) -> str:
        """
        Save edge data to cache.
        
        Args:
            graph_id: Graph identifier
            edges_df: DataFrame with edge data
            
        Returns:
            Cache file path
        """
        cache_file = self.data_dir / f"{graph_id}_edges.csv"
        edges_df.to_csv(cache_file, index=False)
        print(f"[CACHE] Saved data: {cache_file.name} ({len(edges_df)} edges)")
        return str(cache_file)
    
    def load_data_cache(self, graph_id: str) -> Optional[pd.DataFrame]:
        """
        Load edge data from cache.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            DataFrame with edge data or None
        """
        cache_file = self.data_dir / f"{graph_id}_edges.csv"
        
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                print(f"[CACHE] Loaded data: {cache_file.name} ({len(df)} edges)")
                return df
            except Exception as e:
                print(f"[CACHE] Error loading data: {e}")
        
        return None
    
    # =========================================================================
    # Metrics Cache
    # =========================================================================
    
    def save_metrics_cache(
        self, 
        metrics_df: pd.DataFrame,
        version: str = "default"
    ) -> str:
        """
        Save computed metrics to cache.
        
        Args:
            metrics_df: DataFrame with node metrics
            version: Version identifier (v1, v2, etc.)
            
        Returns:
            Cache file path
        """
        cache_file = self.data_dir / f"node_metrics_{version}.csv"
        metrics_df.to_csv(cache_file, index=False)
        print(f"[CACHE] Saved metrics: {cache_file.name} ({len(metrics_df)} nodes)")
        return str(cache_file)
    
    def load_metrics_cache(self, version: str = "default") -> Optional[pd.DataFrame]:
        """
        Load metrics from cache.
        
        Args:
            version: Version identifier
            
        Returns:
            DataFrame with metrics or None
        """
        cache_file = self.data_dir / f"node_metrics_{version}.csv"
        
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                print(f"[CACHE] Loaded metrics: {cache_file.name} ({len(df)} nodes)")
                return df
            except Exception as e:
                print(f"[CACHE] Error loading metrics: {e}")
        
        return None
    
    def get_metrics_hash(self, metrics_df: pd.DataFrame) -> str:
        """Generate hash of metrics dataframe for change detection."""
        content = metrics_df.to_json().encode()
        return hashlib.md5(content).hexdigest()[:12]