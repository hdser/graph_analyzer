"""
Cache Service

Handles caching for layouts, data, metrics, and properties.
All caches use Parquet format for consistency, type preservation, and efficiency.

Layout cache structure:
- cache/layouts/{graph_id}.parquet        - Current working layout (updated incrementally)
- cache/layouts/{graph_id}_base.parquet   - Base layout (from Cytoscape Desktop, protected)
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

from ..config import settings


class CacheService:
    """
    Service for managing layout, data, metrics, and properties caches.
    
    Cache structure:
    - cache/layouts/{graph_id}.parquet (current layout)
    - cache/layouts/{graph_id}_base.parquet (base layout, protected)
    - cache/data/{graph_id}_edges.parquet
    - cache/data/node_metrics_{version}.parquet
    - cache/data/node_properties_{version}.parquet
    """
    
    def __init__(self):
        """Initialize cache service with configured paths."""
        self.layouts_dir = settings.LAYOUTS_DIR
        self.data_dir = settings.DATA_CACHE_DIR
        
        # Ensure directories exist
        self.layouts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Layout Cache (Parquet)
    # =========================================================================
    
    def _positions_to_df(self, positions: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert positions dict to DataFrame."""
        rows = [
            {'node_id': node_id, 'x': pos['x'], 'y': pos['y']}
            for node_id, pos in positions.items()
        ]
        return pd.DataFrame(rows)
    
    def _df_to_positions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert DataFrame to positions dict."""
        return {
            str(row['node_id']): {'x': float(row['x']), 'y': float(row['y'])}
            for _, row in df.iterrows()
        }
    
    def get_cached_layout(self, graph_id: str) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get cached layout for a graph.
        
        Tries to load in order:
        1. Current layout ({graph_id}.parquet)
        2. Base layout ({graph_id}_base.parquet)
        3. Legacy JSON files (for backward compatibility)
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Layout dictionary {node_id: {x, y}} or None if not cached
        """
        # Try current layout first
        current_file = self.layouts_dir / f"{graph_id}.parquet"
        if current_file.exists():
            try:
                df = pd.read_parquet(current_file)
                positions = self._df_to_positions(df)
                print(f"[CACHE] Loaded layout: {current_file.name} ({len(positions)} nodes)")
                return positions
            except Exception as e:
                print(f"[CACHE] Error loading layout: {e}")
        
        # Try base layout
        base_file = self.layouts_dir / f"{graph_id}_base.parquet"
        if base_file.exists():
            try:
                df = pd.read_parquet(base_file)
                positions = self._df_to_positions(df)
                print(f"[CACHE] Loaded base layout: {base_file.name} ({len(positions)} nodes)")
                return positions
            except Exception as e:
                print(f"[CACHE] Error loading base layout: {e}")
        
        # Try legacy JSON files (backward compatibility)
        for json_file in self.layouts_dir.glob(f"{graph_id}*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                positions = data.get('positions', data)
                if positions and isinstance(positions, dict):
                    print(f"[CACHE] Loaded legacy layout: {json_file.name} ({len(positions)} nodes)")
                    return positions
            except Exception:
                continue
        
        return None
    
    def save_layout_cache(
        self, 
        graph_id: str, 
        positions: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Save layout to current cache (overwrites existing).
        
        Args:
            graph_id: Graph identifier
            positions: Layout positions {node_id: {x, y}}
            
        Returns:
            Cache file path
        """
        cache_file = self.layouts_dir / f"{graph_id}.parquet"
        df = self._positions_to_df(positions)
        df.to_parquet(cache_file, index=False)
        print(f"[CACHE] Saved layout: {cache_file.name} ({len(positions)} nodes)")
        return str(cache_file)
    
    def save_base_layout(
        self, 
        graph_id: str, 
        positions: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Save layout as base (protected, from Cytoscape Desktop).
        
        Args:
            graph_id: Graph identifier
            positions: Layout positions {node_id: {x, y}}
            
        Returns:
            Cache file path
        """
        cache_file = self.layouts_dir / f"{graph_id}_base.parquet"
        df = self._positions_to_df(positions)
        df.to_parquet(cache_file, index=False)
        print(f"[CACHE] Saved base layout: {cache_file.name} ({len(positions)} nodes)")
        return str(cache_file)
    
    def has_base_layout(self, graph_id: str) -> bool:
        """Check if a base layout exists for the graph."""
        return (self.layouts_dir / f"{graph_id}_base.parquet").exists()
    
    def list_cached_layouts(self) -> List[Dict[str, Any]]:
        """List all cached layouts with metadata."""
        layouts = []
        
        # List parquet layouts
        for cache_file in self.layouts_dir.glob("*.parquet"):
            try:
                graph_id = cache_file.stem
                is_base = graph_id.endswith('_base')
                if is_base:
                    graph_id = graph_id[:-5]  # Remove '_base' suffix
                
                df = pd.read_parquet(cache_file)
                layouts.append({
                    'filename': cache_file.name,
                    'graph_id': graph_id,
                    'node_count': len(df),
                    'is_base': is_base,
                    'size_mb': cache_file.stat().st_size / (1024 * 1024)
                })
            except Exception:
                continue
        
        return sorted(layouts, key=lambda x: (x.get('graph_id', ''), x.get('is_base', False)))
    
    def clear_layout_cache(self, graph_id: Optional[str] = None, include_base: bool = False):
        """
        Clear layout cache.
        
        Args:
            graph_id: Clear only for this graph (None = clear all)
            include_base: If True, also delete base layouts (default: False)
        """
        if graph_id:
            # Delete current layout
            current_file = self.layouts_dir / f"{graph_id}.parquet"
            if current_file.exists():
                current_file.unlink()
                print(f"[CACHE] Deleted: {current_file.name}")
            
            # Delete base layout only if requested
            if include_base:
                base_file = self.layouts_dir / f"{graph_id}_base.parquet"
                if base_file.exists():
                    base_file.unlink()
                    print(f"[CACHE] Deleted: {base_file.name}")
            
            # Delete legacy JSON files
            for json_file in self.layouts_dir.glob(f"{graph_id}*.json"):
                json_file.unlink()
                print(f"[CACHE] Deleted legacy: {json_file.name}")
        else:
            # Clear all
            for cache_file in self.layouts_dir.glob("*.parquet"):
                if not include_base and cache_file.stem.endswith('_base'):
                    continue
                cache_file.unlink()
                print(f"[CACHE] Deleted: {cache_file.name}")
            
            # Delete all legacy JSON files
            for json_file in self.layouts_dir.glob("*.json"):
                json_file.unlink()
                print(f"[CACHE] Deleted legacy: {json_file.name}")
    
    # =========================================================================
    # Data Cache (Parquet - for edge DataFrames)
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
        cache_file = self.data_dir / f"{graph_id}_edges.parquet"
        edges_df.to_parquet(cache_file, index=False)
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
        cache_file = self.data_dir / f"{graph_id}_edges.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                print(f"[CACHE] Loaded data: {cache_file.name} ({len(df)} edges)")
                return df
            except Exception as e:
                print(f"[CACHE] Error loading data: {e}")
        
        return None
    
    # =========================================================================
    # Metrics Cache (Parquet)
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
        cache_file = self.data_dir / f"node_metrics_{version}.parquet"
        metrics_df.to_parquet(cache_file, index=False)
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
        cache_file = self.data_dir / f"node_metrics_{version}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                print(f"[CACHE] Loaded metrics: {cache_file.name} ({len(df)} nodes)")
                return df
            except Exception as e:
                print(f"[CACHE] Error loading metrics: {e}")
        
        return None
    
    def get_metrics_hash(self, metrics_df: pd.DataFrame) -> str:
        """Generate hash of metrics dataframe for change detection."""
        content = metrics_df.to_json().encode()
        return hashlib.md5(content).hexdigest()[:12]
    
    # =========================================================================
    # Properties Cache (Parquet)
    # =========================================================================
    
    def save_properties_cache(
        self, 
        properties_df: pd.DataFrame,
        version: str = "default"
    ) -> str:
        """
        Save node properties to cache.
        
        Args:
            properties_df: DataFrame with node properties
            version: Version identifier (v1, v2, etc.)
            
        Returns:
            Cache file path
        """
        cache_file = self.data_dir / f"node_properties_{version}.parquet"
        properties_df.to_parquet(cache_file, index=False)
        print(f"[CACHE] Saved properties: {cache_file.name} ({len(properties_df)} nodes)")
        return str(cache_file)
    
    def load_properties_cache(self, version: str = "default") -> Optional[pd.DataFrame]:
        """
        Load properties from cache.
        
        Args:
            version: Version identifier
            
        Returns:
            DataFrame with properties or None
        """
        cache_file = self.data_dir / f"node_properties_{version}.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                print(f"[CACHE] Loaded properties: {cache_file.name} ({len(df)} nodes)")
                return df
            except Exception as e:
                print(f"[CACHE] Error loading properties: {e}")
        
        return None