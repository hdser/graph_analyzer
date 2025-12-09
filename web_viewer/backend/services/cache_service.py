"""
Cache Service

Handles caching for layouts, data, metrics, and properties.
All caches use Parquet format for consistency, type preservation, and efficiency.

Layout cache structure:
- cache/layouts/{graph_id}.parquet        - Current working layout (updated incrementally)
- cache/layouts/{graph_id}_base.parquet   - Base layout (from Cytoscape Desktop, protected)

API properties cache structure:
- cache/data/api_properties_{provider}_{version}.parquet
- cache/data/api_properties_{provider}_{version}.meta.json (timestamp for TTL)
"""

import json
import hashlib
import time
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
    - cache/data/api_properties_{provider}_{version}.parquet
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
    # Properties Cache (Parquet - SQL-based properties)
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
    
    # =========================================================================
    # API Properties Cache (Parquet with TTL)
    # =========================================================================
    
    def _get_api_cache_file(self, provider: str, version: str) -> Path:
        """Get cache file path for API properties."""
        return self.data_dir / f"api_properties_{provider}_{version}.parquet"
    
    def _get_api_meta_file(self, provider: str, version: str) -> Path:
        """Get metadata file path for API properties cache."""
        return self.data_dir / f"api_properties_{provider}_{version}.meta.json"
    
    def save_api_properties_cache(
        self,
        provider: str,
        version: str,
        properties_df: pd.DataFrame
    ) -> str:
        """
        Save API properties to cache with timestamp metadata.
        
        Args:
            provider: Provider name (e.g., 'blacklist')
            version: Version identifier (e.g., 'v2')
            properties_df: DataFrame with properties
            
        Returns:
            Cache file path
        """
        cache_file = self._get_api_cache_file(provider, version)
        meta_file = self._get_api_meta_file(provider, version)
        
        # Save data
        properties_df.to_parquet(cache_file, index=False)
        
        # Save metadata with timestamp
        meta = {
            'provider': provider,
            'version': version,
            'timestamp': time.time(),
            'row_count': len(properties_df),
            'columns': list(properties_df.columns)
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        
        print(f"[CACHE] Saved API properties: {cache_file.name} ({len(properties_df)} rows)")
        return str(cache_file)
    
    def load_api_properties_cache(
        self,
        provider: str,
        version: str,
        ttl: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load API properties from cache if not expired.
        
        Args:
            provider: Provider name
            version: Version identifier
            ttl: Time-to-live in seconds (None = use settings default, 0 = no expiry)
            
        Returns:
            DataFrame with properties or None if cache miss/expired
        """
        cache_file = self._get_api_cache_file(provider, version)
        meta_file = self._get_api_meta_file(provider, version)
        
        if not cache_file.exists():
            return None
        
        # Check TTL if metadata exists
        if ttl is None:
            ttl = settings.EXTERNAL_API_CACHE_TTL
        
        if ttl > 0 and meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                
                cached_time = meta.get('timestamp', 0)
                age = time.time() - cached_time
                
                if age > ttl:
                    print(f"[CACHE] API cache expired for {provider}/{version} "
                          f"(age: {age:.0f}s, ttl: {ttl}s)")
                    return None
                
                print(f"[CACHE] API cache valid for {provider}/{version} "
                      f"(age: {age:.0f}s, ttl: {ttl}s)")
                
            except Exception as e:
                print(f"[CACHE] Error reading API cache metadata: {e}")
                # Continue to try loading data without TTL check
        
        try:
            df = pd.read_parquet(cache_file)
            print(f"[CACHE] Loaded API properties: {cache_file.name} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"[CACHE] Error loading API properties: {e}")
            return None
    
    def is_api_cache_valid(
        self,
        provider: str,
        version: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Check if API cache exists and is not expired.
        
        Args:
            provider: Provider name
            version: Version identifier
            ttl: Time-to-live in seconds
            
        Returns:
            True if cache is valid, False otherwise
        """
        cache_file = self._get_api_cache_file(provider, version)
        meta_file = self._get_api_meta_file(provider, version)
        
        if not cache_file.exists():
            return False
        
        if ttl is None:
            ttl = settings.EXTERNAL_API_CACHE_TTL
        
        if ttl == 0:
            return True
        
        if not meta_file.exists():
            return False
        
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            cached_time = meta.get('timestamp', 0)
            age = time.time() - cached_time
            return age <= ttl
            
        except Exception:
            return False
    
    def clear_api_properties_cache(
        self,
        provider: Optional[str] = None,
        version: Optional[str] = None
    ):
        """
        Clear API properties cache.
        
        Args:
            provider: Clear only this provider (None = all)
            version: Clear only this version (None = all)
        """
        pattern = "api_properties_"
        if provider:
            pattern += f"{provider}_"
        if version:
            pattern += f"{version}"
        pattern += "*"
        
        for cache_file in self.data_dir.glob(pattern):
            cache_file.unlink()
            print(f"[CACHE] Deleted API cache: {cache_file.name}")
    
    def list_api_properties_caches(self) -> List[Dict[str, Any]]:
        """List all API properties caches with metadata."""
        caches = []
        
        for cache_file in self.data_dir.glob("api_properties_*.parquet"):
            try:
                # Parse filename: api_properties_{provider}_{version}.parquet
                parts = cache_file.stem.split('_')
                if len(parts) >= 4:
                    provider = parts[2]
                    version = parts[3]
                else:
                    provider = "unknown"
                    version = "unknown"
                
                # Get metadata if available
                meta_file = cache_file.with_suffix('.meta.json')
                meta = {}
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                
                df = pd.read_parquet(cache_file)
                caches.append({
                    'filename': cache_file.name,
                    'provider': provider,
                    'version': version,
                    'row_count': len(df),
                    'columns': list(df.columns),
                    'timestamp': meta.get('timestamp'),
                    'age_seconds': time.time() - meta.get('timestamp', 0) if meta.get('timestamp') else None,
                    'size_mb': cache_file.stat().st_size / (1024 * 1024)
                })
            except Exception:
                continue
        
        return sorted(caches, key=lambda x: (x.get('provider', ''), x.get('version', '')))