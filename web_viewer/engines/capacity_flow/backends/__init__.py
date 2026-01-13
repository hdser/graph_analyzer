"""
Flow Backends

Backend implementations for max flow computation.

Location: web_viewer/engines/capacity_flow/backends/__init__.py
"""
import logging
from typing import Dict, List, Optional, Any, Type

from .base import BaseFlowBackend
from .networkx_backend import NetworkXBackend
from .ortools_backend import ORToolsBackend

logger = logging.getLogger(__name__)

# Registry of available backends
_BACKENDS: Dict[str, Type[BaseFlowBackend]] = {
    "networkx": NetworkXBackend,
    "ortools": ORToolsBackend,
}

# Priority order for auto-selection
_BACKEND_PRIORITY = ["ortools", "networkx"]


def get_backend(name: str) -> Optional[BaseFlowBackend]:
    """
    Get backend instance by name.
    
    Args:
        name: Backend name
        
    Returns:
        Backend instance or None if not available
    """
    backend_class = _BACKENDS.get(name)
    if backend_class is None:
        logger.warning(f"Unknown backend: {name}")
        return None
    
    if not backend_class.is_available():
        logger.warning(f"Backend {name} is not available")
        return None
    
    return backend_class()


def get_available_backends() -> List[str]:
    """Get list of available backend names."""
    return [
        name for name, cls in _BACKENDS.items()
        if cls.is_available()
    ]


def get_best_backend() -> Optional[BaseFlowBackend]:
    """
    Get best available backend based on priority.
    
    Returns:
        Backend instance or None if none available
    """
    for name in _BACKEND_PRIORITY:
        backend_class = _BACKENDS.get(name)
        if backend_class and backend_class.is_available():
            return backend_class()
    
    return None


def get_backend_info() -> List[Dict[str, Any]]:
    """Get information about all backends."""
    return [
        cls.get_info()
        for cls in _BACKENDS.values()
    ]


def register_backend(name: str, backend_class: Type[BaseFlowBackend]) -> None:
    """Register a custom backend."""
    _BACKENDS[name] = backend_class


__all__ = [
    "BaseFlowBackend",
    "NetworkXBackend",
    "ORToolsBackend",
    "get_backend",
    "get_available_backends",
    "get_best_backend",
    "get_backend_info",
    "register_backend",
]