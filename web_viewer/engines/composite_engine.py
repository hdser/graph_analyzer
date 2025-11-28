"""
Composite Metric Engine

Engine for creating composite metrics from existing metrics.

Supports:
- Binary operations: multiply, add, subtract, divide, max, min, average
- Normalization of inputs
- Persistence to disk cache
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


class CompositeMetricEngine:
    """
    Engine for creating composite metrics from existing metrics.
    
    Supports:
    - Binary operations: multiply, add, subtract, divide, max, min, average
    - Normalization of inputs
    - Persistence to disk cache
    """
    
    OPERATIONS = {
        "multiply": {
            "symbol": "x",
            "description": "Multiply two metrics",
            "func": lambda a, b: a * b
        },
        "add": {
            "symbol": "+",
            "description": "Add two metrics",
            "func": lambda a, b: a + b
        },
        "subtract": {
            "symbol": "-",
            "description": "Subtract second metric from first",
            "func": lambda a, b: a - b
        },
        "divide": {
            "symbol": "/",
            "description": "Divide first metric by second (with epsilon to avoid division by zero)",
            "func": lambda a, b: a / (b + 1e-10)
        },
        "maximum": {
            "symbol": "max",
            "description": "Maximum of two metrics",
            "func": lambda a, b: np.maximum(a, b)
        },
        "minimum": {
            "symbol": "min",
            "description": "Minimum of two metrics",
            "func": lambda a, b: np.minimum(a, b)
        },
        "average": {
            "symbol": "avg",
            "description": "Average of two metrics",
            "func": lambda a, b: (a + b) / 2
        },
        "weighted_sum": {
            "symbol": "weighted",
            "description": "Weighted sum of metrics",
            "func": lambda a, b, w1=0.5, w2=0.5: w1 * a + w2 * b
        },
        "norm_multiply": {
            "symbol": "norm_x",
            "description": "Multiply normalized metrics (scale-independent)",
            "func": lambda a, b: a * b  # Normalization applied before
        }
    }

    def __init__(self, cache_path: str = "cache/composite_metrics.json"):
        """
        Initialize composite metric engine.
        
        Args:
            cache_path: Path to JSON file for persisting composites
        """
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._composites: List[Dict[str, Any]] = []
        self._load_cache()
    
    def _load_cache(self):
        """Load persisted composite metrics from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                    self._composites = data.get('metrics', [])
                print(f"[COMPOSITE] Loaded {len(self._composites)} saved composites")
            except Exception as e:
                print(f"[COMPOSITE] Error loading cache: {e}")
                self._composites = []
        else:
            self._composites = []
    
    def _save_cache(self):
        """Save composite metrics to disk."""
        try:
            data = {
                'metrics': self._composites,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[COMPOSITE] Saved {len(self._composites)} composites to cache")
        except Exception as e:
            print(f"[COMPOSITE] Error saving cache: {e}")
    
    @classmethod
    def get_available_operations(cls) -> Dict[str, Any]:
        """Get available operations for frontend."""
        return {
            name: {
                'symbol': info['symbol'],
                'description': info['description']
            }
            for name, info in cls.OPERATIONS.items()
        }
    
    def _normalize_metric(self, values: np.ndarray) -> np.ndarray:
        """Normalize metric values to [0, 1] range."""
        values = np.array(values, dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0)
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val - min_val < 1e-10:
            return np.zeros_like(values)
        
        return (values - min_val) / (max_val - min_val)
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID for composite metric."""
        timestamp = int(time.time())
        hash_part = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"cm_{timestamp}_{hash_part}"
    
    def create_composite(
        self,
        df: pd.DataFrame,
        name: str,
        metrics: List[str],
        operation: str,
        weights: Optional[List[float]] = None,
        normalize: bool = False,
        save: bool = True,
        version: str = "default"
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Create a composite metric from existing metrics.
        
        Args:
            df: DataFrame with metric columns
            name: Name for new metric
            metrics: List of source metric names (usually 2)
            operation: Operation name from OPERATIONS
            weights: Weights for weighted operations
            normalize: Whether to normalize inputs first
            save: Whether to persist to cache
            version: Graph version identifier
            
        Returns:
            Tuple of (metric_series, metadata_dict)
        """
        # Validate operation
        if operation not in self.OPERATIONS:
            raise ValueError(f"Unknown operation: {operation}. Available: {list(self.OPERATIONS.keys())}")
        
        # Validate metrics
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise ValueError(f"Metrics not found: {missing}")
        
        if len(metrics) < 2:
            raise ValueError("At least 2 metrics required for composite")
        
        # Get values
        values = [df[m].values.astype(np.float64) for m in metrics]
        
        # Handle NaN
        values = [np.nan_to_num(v, nan=0.0) for v in values]
        
        # Normalize if requested
        if normalize or operation == 'norm_multiply':
            values = [self._normalize_metric(v) for v in values]
        
        # Apply operation
        op_func = self.OPERATIONS[operation]['func']
        
        if operation == 'weighted_sum' and weights and len(weights) >= 2:
            result = op_func(values[0], values[1], weights[0], weights[1])
        else:
            result = op_func(values[0], values[1])
        
        # Create Series
        if 'avatar' in df.columns:
            result_series = pd.Series(result, index=df['avatar'])
        else:
            result_series = pd.Series(result, index=df.index)
        
        # Build formula string
        symbol = self.OPERATIONS[operation]['symbol']
        formula = f"{metrics[0]} {symbol} {metrics[1]}"
        if normalize:
            formula = f"norm({metrics[0]}) {symbol} norm({metrics[1]})"
        
        # Metadata
        metadata = {
            'name': name,
            'formula': formula,
            'operation': operation,
            'source_metrics': metrics,
            'weights': weights,
            'normalize': normalize,
            'version': version,
            'statistics': {
                'min': float(np.min(result)),
                'max': float(np.max(result)),
                'mean': float(np.mean(result)),
                'std': float(np.std(result)),
                'median': float(np.median(result))
            }
        }
        
        # Save to cache if requested
        if save:
            composite_id = self._generate_id(name)
            composite_record = {
                'id': composite_id,
                'name': name,
                'formula': formula,
                'operation': operation,
                'source_metrics': metrics,
                'weights': weights,
                'normalize': normalize,
                'created_at': datetime.now().isoformat(),
                'version': version
            }
            
            # Remove existing with same name and version
            self._composites = [
                c for c in self._composites 
                if not (c['name'] == name and c.get('version', 'default') == version)
            ]
            
            self._composites.append(composite_record)
            self._save_cache()
            
            metadata['id'] = composite_id
            metadata['saved'] = True
        else:
            metadata['saved'] = False
        
        return result_series, metadata
    
    def get_saved_composites(self, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of saved composite metrics.
        
        Args:
            version: Optional version filter
            
        Returns:
            List of composite metric records
        """
        if version:
            return [c for c in self._composites if c.get('version', 'default') == version]
        return self._composites.copy()
    
    def get_composite_by_id(self, composite_id: str) -> Optional[Dict[str, Any]]:
        """Get a composite by its ID."""
        for c in self._composites:
            if c['id'] == composite_id:
                return c
        return None
    
    def delete_composite(self, composite_id: str) -> bool:
        """
        Delete a saved composite metric by ID.
        
        Args:
            composite_id: ID of composite to delete
            
        Returns:
            True if deleted, False if not found
        """
        original_len = len(self._composites)
        self._composites = [c for c in self._composites if c['id'] != composite_id]
        
        if len(self._composites) < original_len:
            self._save_cache()
            return True
        return False
    
    def delete_composite_by_name(self, name: str, version: str = None) -> bool:
        """
        Delete a saved composite metric by name.
        
        Args:
            name: Name of composite to delete
            version: Optional version filter
            
        Returns:
            True if deleted, False if not found
        """
        original_len = len(self._composites)
        if version:
            self._composites = [
                c for c in self._composites 
                if not (c['name'] == name and c.get('version', 'default') == version)
            ]
        else:
            self._composites = [c for c in self._composites if c['name'] != name]
        
        if len(self._composites) < original_len:
            self._save_cache()
            return True
        return False
    
    def apply_saved_composite(
        self, 
        composite_id: str, 
        df: pd.DataFrame
    ) -> Tuple[Optional[pd.Series], Optional[Dict[str, Any]]]:
        """
        Apply a saved composite metric formula to new data.
        
        Args:
            composite_id: ID of saved composite
            df: DataFrame with required source metrics
            
        Returns:
            Tuple of (metric_series, metadata) or (None, None) if not found
        """
        composite = self.get_composite_by_id(composite_id)
        if not composite:
            return None, None
        
        return self.create_composite(
            df=df,
            name=composite['name'],
            metrics=composite['source_metrics'],
            operation=composite['operation'],
            weights=composite.get('weights'),
            normalize=composite.get('normalize', False),
            save=False,  # Don't re-save
            version=composite.get('version', 'default')
        )