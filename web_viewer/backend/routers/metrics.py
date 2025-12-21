"""
Metrics Router

API endpoints for metrics computation and discovery.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from ..models.requests import MetricsConfig
from ..services.network_service import network_service
from engines.metrics import (
    METRIC_REGISTRY,
    METRIC_CATEGORIES,
    METRIC_PRESETS,
    MetricResolver,
    list_all_metrics,
    list_categories,
    list_presets,
    get_metric,
    get_category_metrics,
    get_preset_metrics,
)


router = APIRouter(prefix="/api", tags=["metrics"])


@router.post("/metrics")
def update_metrics(config: MetricsConfig) -> Dict[str, Any]:
    """
    Compute/update metrics for loaded graphs.
    
    Supports granular metric selection via:
    - preset: Named preset (basic, essential, moderate, comprehensive, all, trust_analysis, influence, structure)
    - categories: List of category names
    - metrics: List of individual metric names
    - exclude_metrics: Metrics to exclude
    
    Examples:
    ```json
    // Individual metrics
    {"metrics": ["pagerank", "eigentrust", "betweenness_centrality"]}
    
    // Categories
    {"categories": ["centrality", "trust"]}
    
    // Preset
    {"preset": "essential"}
    
    // Category with exclusions
    {"categories": ["centrality"], "exclude_metrics": ["katz_centrality"]}
    ```
    
    Returns:
        Dictionary with computed metrics list, computation time, and node data
    """
    try:
        result = network_service.update_metrics(config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error updating metrics: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
def list_metrics() -> Dict[str, Any]:
    """
    List all available metrics with metadata.
    
    Returns:
        Dictionary with:
        - metrics: List of all available metrics
        - categories: List of categories with their metrics
        - presets: List of presets with descriptions
        - total_count: Total number of available metrics
    """
    return {
        "metrics": list_all_metrics(),
        "categories": list_categories(),
        "presets": list_presets(),
        "total_count": len(METRIC_REGISTRY),
    }


@router.get("/metrics/info/{metric_name}")
def get_metric_info(metric_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific metric.
    
    Args:
        metric_name: Name of the metric
        
    Returns:
        Metric definition with all metadata
    """
    metric = get_metric(metric_name)
    if not metric:
        raise HTTPException(status_code=404, detail=f"Metric not found: {metric_name}")
    
    return {
        "name": metric.name,
        "category": metric.category,
        "description": metric.description,
        "cost": metric.cost,
        "max_nodes": metric.max_nodes,
        "graph_type": metric.graph_type,
        "requires_connected": metric.requires_connected,
        "output_columns": metric.output_columns,
        "dependencies": metric.dependencies,
        "source": metric.source,
        "citation": metric.citation,
    }


@router.get("/metrics/categories")
def list_metric_categories() -> List[Dict[str, Any]]:
    """
    List all metric categories with descriptions.
    
    Returns:
        List of categories with their metrics
    """
    return list_categories()


@router.get("/metrics/categories/{category_name}")
def get_category_info(category_name: str) -> Dict[str, Any]:
    """
    Get information about a specific category.
    
    Args:
        category_name: Name of the category
        
    Returns:
        Category info with list of metrics
    """
    if category_name not in METRIC_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category not found: {category_name}")
    
    cat = METRIC_CATEGORIES[category_name]
    return {
        "name": category_name,
        "description": cat["description"],
        "metrics": cat["metrics"],
        "metric_count": len(cat["metrics"]),
    }


@router.get("/metrics/presets")
def list_metric_presets() -> List[Dict[str, Any]]:
    """
    List all metric presets with descriptions.
    
    Returns:
        List of presets with their configurations
    """
    return list_presets()


@router.get("/metrics/presets/{preset_name}")
def get_preset_info(preset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Preset info with list of metrics it includes
    """
    if preset_name not in METRIC_PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_name}")
    
    preset = METRIC_PRESETS[preset_name]
    metrics = get_preset_metrics(preset_name)
    
    return {
        "name": preset_name,
        "description": preset["description"],
        "categories": preset.get("categories", []),
        "explicit_metrics": preset.get("metrics", []),
        "all_metrics": metrics,
        "metric_count": len(metrics),
    }


@router.post("/metrics/preview")
def preview_metrics(config: MetricsConfig) -> Dict[str, Any]:
    """
    Preview what metrics would be computed without running computation.
    
    Useful for validating configuration before running expensive computations.
    
    Args:
        config: MetricsConfig with selection parameters
        
    Returns:
        Dictionary with:
        - total_metrics: Number of metrics that would be computed
        - by_category: Metrics grouped by category
        - metrics: List of metric names with cost info
        - estimated_cost: Overall estimated computation cost
    """
    resolver = MetricResolver()
    
    try:
        info = resolver.get_metrics_info(
            preset=config.preset,
            categories=config.categories,
            metrics=config.metrics,
        )
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics/search")
def search_metrics(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    max_cost: Optional[str] = Query(None, description="Maximum cost: low, medium, high, very_high"),
) -> List[Dict[str, Any]]:
    """
    Search available metrics by name, description, or category.
    
    Args:
        q: Search query string
        category: Optional category filter
        max_cost: Optional maximum cost filter
        
    Returns:
        List of matching metrics
    """
    cost_order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    max_cost_value = cost_order.get(max_cost, 4) if max_cost else 4
    
    results = []
    query_lower = q.lower()
    
    for name, metric in METRIC_REGISTRY.items():
        # Category filter
        if category and metric.category != category:
            continue
        
        # Cost filter
        if cost_order.get(metric.cost, 0) > max_cost_value:
            continue
        
        # Search in name, description, category
        if (query_lower in name.lower() or 
            query_lower in metric.description.lower() or
            query_lower in metric.category.lower()):
            results.append({
                "name": metric.name,
                "category": metric.category,
                "description": metric.description,
                "cost": metric.cost,
                "output_columns": metric.output_columns,
            })
    
    return results