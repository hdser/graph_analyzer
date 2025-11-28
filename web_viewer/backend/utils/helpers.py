"""
Helper Functions

Shared utility functions used across the application.
"""

from typing import Any, Dict, List

import numpy as np


def clean_numpy_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        data: Dictionary that may contain numpy types
        
    Returns:
        Dictionary with all numpy types converted to Python native types
    """
    clean_data = {}
    for k, v in data.items():
        if isinstance(v, (np.integer,)):
            clean_data[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean_data[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean_data[k] = v.tolist()
        elif isinstance(v, dict):
            clean_data[k] = clean_numpy_types(v)
        elif isinstance(v, list):
            clean_data[k] = [
                clean_numpy_types(item) if isinstance(item, dict) else item 
                for item in v
            ]
        else:
            clean_data[k] = v
    return clean_data


def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number for display.
    
    Args:
        value: Number to format
        precision: Decimal places
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{precision}f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]