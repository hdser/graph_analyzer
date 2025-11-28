"""
Parallel Execution Utilities for Anomaly Detection

Provides parallelization for:
- Group-aware detection (parallel per-group processing)
- Large-scale scoring (parallel batch scoring)
"""

import os
from typing import List, Dict, Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
import pandas as pd


def get_optimal_workers(n_jobs: int = -1) -> int:
    """
    Get optimal number of workers.
    
    Args:
        n_jobs: Number of jobs. -1 means use all CPUs.
        
    Returns:
        Number of workers to use
    """
    if n_jobs == -1:
        return max(1, multiprocessing.cpu_count() - 1)
    elif n_jobs == 0:
        return 1
    else:
        return min(n_jobs, multiprocessing.cpu_count())


class ParallelExecutor:
    """
    Parallel execution manager for anomaly detection.
    
    Supports both thread-based and process-based parallelism.
    """
    
    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "threading",
        verbose: bool = False,
    ):
        """
        Initialize parallel executor.
        
        Args:
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            backend: "threading" or "multiprocessing"
            verbose: Print progress information
        """
        self.n_workers = get_optimal_workers(n_jobs)
        self.backend = backend
        self.verbose = verbose
    
    def map_groups(
        self,
        func: Callable,
        groups: List[Tuple[Any, pd.DataFrame]],
        **kwargs,
    ) -> Dict[Any, Any]:
        """
        Apply function to each group in parallel.
        
        Args:
            func: Function to apply. Signature: func(group_df, **kwargs) -> result
            groups: List of (group_key, group_dataframe) tuples
            **kwargs: Additional arguments passed to func
            
        Returns:
            Dictionary mapping group_key to result
        """
        if len(groups) == 0:
            return {}
        
        # For single group or single worker, run sequentially
        if len(groups) == 1 or self.n_workers == 1:
            results = {}
            for group_key, group_df in groups:
                results[group_key] = func(group_df, **kwargs)
            return results
        
        # Choose executor based on backend
        ExecutorClass = (
            ThreadPoolExecutor if self.backend == "threading" 
            else ProcessPoolExecutor
        )
        
        results = {}
        
        with ExecutorClass(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_key = {}
            for group_key, group_df in groups:
                future = executor.submit(func, group_df, **kwargs)
                future_to_key[future] = group_key
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_key):
                group_key = future_to_key[future]
                try:
                    results[group_key] = future.result()
                    completed += 1
                    if self.verbose:
                        print(f"[Parallel] Completed {completed}/{len(groups)} groups")
                except Exception as e:
                    if self.verbose:
                        print(f"[Parallel] Error in group {group_key}: {e}")
                    results[group_key] = None
        
        return results
    
    def parallel_score(
        self,
        score_func: Callable,
        X_chunks: List[np.ndarray],
        model: Any,
    ) -> np.ndarray:
        """
        Score multiple chunks in parallel.
        
        Args:
            score_func: Function to score a chunk. Signature: score_func(model, X) -> scores
            X_chunks: List of data chunks
            model: Fitted model to use for scoring
            
        Returns:
            Concatenated scores array
        """
        if len(X_chunks) == 0:
            return np.array([])
        
        if len(X_chunks) == 1:
            return score_func(model, X_chunks[0])
        
        # For scoring, threading is usually fine (GIL released during numpy ops)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(score_func, model, chunk)
                for chunk in X_chunks
            ]
            
            results = [future.result() for future in futures]
        
        return np.concatenate(results)


def chunk_array(arr: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """
    Split array into chunks.
    
    Args:
        arr: Input array
        chunk_size: Size of each chunk
        
    Returns:
        List of array chunks
    """
    n_chunks = (len(arr) + chunk_size - 1) // chunk_size
    return [arr[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Split DataFrame into chunks.
    
    Args:
        df: Input DataFrame
        chunk_size: Size of each chunk
        
    Returns:
        List of DataFrame chunks
    """
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    return [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n_chunks)]