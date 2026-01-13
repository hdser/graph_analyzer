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
def compute_metrics(config: MetricsConfig) -> Dict[str, Any]:
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
    
    // Preset with exclusions
    {"preset": "comprehensive", "exclude_metrics": ["katz_centrality"]}
    
    // Combined (all sources merged)
    {"preset": "basic", "metrics": ["eigentrust"], "categories": ["trust"]}
    ```
    """
    result = network_service.update_metrics(config)
    return result


@router.get("/metrics")
def get_available_metrics() -> Dict[str, Any]:
    """
    Get all available metrics organized by category.
    
    Returns complete metric catalog with:
    - All metrics with metadata
    - Categories with their metrics
    - Available presets
    - igraph availability status
    """
    # Check igraph availability
    igraph_available = False
    try:
        import igraph
        igraph_available = True
    except ImportError:
        pass
    
    # Get metrics by cost level
    metrics_by_cost = {"low": [], "medium": [], "high": [], "very_high": []}
    for name, metric in METRIC_REGISTRY.items():
        cost = getattr(metric, 'cost', 'medium')
        if cost in metrics_by_cost:
            metrics_by_cost[cost].append(name)
    
    return {
        "metrics": list_all_metrics(),
        "categories": list_categories(),
        "presets": list_presets(),
        "total_count": len(METRIC_REGISTRY),
        "igraph_available": igraph_available,
        "metrics_by_cost": metrics_by_cost,
    }


@router.get("/metrics/info/{metric_name}")
def get_metric_info(metric_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific metric.
    
    Returns:
    - Full metric definition
    - Parameter definitions if configurable
    - Dependencies
    - Availability status
    """
    metric = get_metric(metric_name)
    if not metric:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    # Check if metric requires igraph
    requires_igraph = getattr(metric, 'requires_igraph', False)
    igraph_available = False
    try:
        import igraph
        igraph_available = True
    except ImportError:
        pass
    
    # Build parameter info
    parameters = []
    if hasattr(metric, 'parameters') and metric.parameters:
        for param in metric.parameters:
            parameters.append({
                'name': param.name,
                'type': param.type,
                'default': param.default,
                'description': param.description,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'choices': param.choices,
                'step': param.step,
            })
    
    return {
        "name": metric.name,
        "category": metric.category,
        "description": metric.description,
        "algorithm_class": metric.algorithm_class,
        "graph_type": metric.graph_type,
        "dependencies": metric.dependencies,
        "cost": metric.cost,
        "max_nodes": metric.max_nodes,
        "requires_connected": metric.requires_connected,
        "output_columns": metric.output_columns,
        "source": metric.source,
        "citation": metric.citation,
        "parameters": parameters,
        "requires_igraph": requires_igraph,
        "available": not requires_igraph or igraph_available,
    }


@router.get("/metrics/parameters/{metric_name}")
def get_metric_parameters(metric_name: str) -> Dict[str, Any]:
    """
    Get parameter definitions for a configurable metric.
    
    Returns parameter schema for UI rendering.
    """
    metric = get_metric(metric_name)
    if not metric:
        raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
    
    parameters = []
    if hasattr(metric, 'parameters') and metric.parameters:
        for param in metric.parameters:
            parameters.append({
                'name': param.name,
                'type': param.type,
                'default': param.default,
                'description': param.description,
                'min_value': param.min_value,
                'max_value': param.max_value,
                'choices': param.choices,
                'step': param.step,
            })
    
    return {
        "metric": metric_name,
        "configurable": len(parameters) > 0,
        "parameters": parameters,
    }


@router.get("/metrics/categories")
def get_categories() -> Dict[str, Any]:
    """
    Get all metric categories with their metrics.
    """
    return {
        "categories": list_categories(),
        "count": len(METRIC_CATEGORIES),
    }


@router.get("/metrics/categories/{category_name}")
def get_category(category_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific category.
    """
    if category_name not in METRIC_CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category '{category_name}' not found")
    
    cat = METRIC_CATEGORIES[category_name]
    metrics_info = []
    
    for metric_name in cat['metrics']:
        metric = get_metric(metric_name)
        if metric:
            metrics_info.append({
                'name': metric.name,
                'description': metric.description,
                'cost': metric.cost,
                'max_nodes': metric.max_nodes,
            })
    
    return {
        "name": category_name,
        "description": cat['description'],
        "metrics": cat['metrics'],
        "metrics_info": metrics_info,
        "metric_count": len(cat['metrics']),
    }


@router.get("/metrics/presets")
def get_presets() -> Dict[str, Any]:
    """
    Get all available metric presets.
    """
    return {
        "presets": list_presets(),
        "count": len(METRIC_PRESETS),
    }


@router.get("/metrics/presets/{preset_name}")
def get_preset(preset_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific preset.
    """
    if preset_name not in METRIC_PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    
    preset = METRIC_PRESETS[preset_name]
    metrics = get_preset_metrics(preset_name)
    
    return {
        "name": preset_name,
        "description": preset['description'],
        "categories": preset.get('categories', []),
        "metrics": metrics,
        "metric_count": len(metrics),
    }


@router.post("/metrics/preview")
def preview_metrics(config: MetricsConfig) -> Dict[str, Any]:
    """
    Preview which metrics would be computed without actually computing them.
    
    Useful for understanding what a preset/category selection includes.
    """
    resolver = MetricResolver()
    resolved = resolver.resolve(
        preset=config.preset,
        categories=config.categories,
        metrics=config.metrics,
        exclude_metrics=config.exclude_metrics,
    )
    
    # Organize by category
    by_category = {}
    for metric_name in resolved:
        metric = get_metric(metric_name)
        if metric:
            cat = metric.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                'name': metric_name,
                'cost': metric.cost,
                'max_nodes': metric.max_nodes,
            })
    
    # Count by cost
    cost_counts = {"low": 0, "medium": 0, "high": 0, "very_high": 0}
    for metric_name in resolved:
        metric = get_metric(metric_name)
        if metric:
            cost = getattr(metric, 'cost', 'medium')
            if cost in cost_counts:
                cost_counts[cost] += 1
    
    return {
        "metrics": list(resolved),
        "by_category": by_category,
        "total_count": len(resolved),
        "cost_breakdown": cost_counts,
        "effective_metric_count": len(resolved),
    }


@router.get("/metrics/igraph")
def get_igraph_status() -> Dict[str, Any]:
    """
    Get igraph availability status and igraph-specific metrics.
    """
    igraph_available = False
    igraph_version = None
    
    try:
        import igraph
        igraph_available = True
        igraph_version = igraph.__version__
    except ImportError:
        pass
    
    # Find metrics that require igraph
    igraph_metrics = []
    for name, metric in METRIC_REGISTRY.items():
        if getattr(metric, 'requires_igraph', False):
            igraph_metrics.append(name)
    
    return {
        "available": igraph_available,
        "version": igraph_version,
        "igraph_metrics": igraph_metrics,
        "igraph_metrics_count": len(igraph_metrics),
    }


@router.get("/metrics/by-cost/{cost}")
def get_metrics_by_cost(cost: str) -> Dict[str, Any]:
    """
    Get metrics filtered by cost level.
    
    Cost levels: low, medium, high, very_high
    """
    valid_costs = ["low", "medium", "high", "very_high"]
    if cost not in valid_costs:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid cost level. Must be one of: {valid_costs}"
        )
    
    metrics = []
    for name, metric in METRIC_REGISTRY.items():
        if getattr(metric, 'cost', 'medium') == cost:
            metrics.append({
                'name': name,
                'category': metric.category,
                'description': metric.description,
                'max_nodes': metric.max_nodes,
            })
    
    return {
        "cost": cost,
        "metrics": metrics,
        "count": len(metrics),
    }


@router.get("/metrics/search")
def search_metrics(
    q: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    cost: Optional[str] = Query(None, description="Filter by cost level"),
) -> Dict[str, Any]:
    """
    Search metrics by name or description.
    
    Supports optional filtering by category and cost level.
    """
    q_lower = q.lower()
    results = []
    
    for name, metric in METRIC_REGISTRY.items():
        # Check if query matches name or description
        if q_lower in name.lower() or q_lower in metric.description.lower():
            # Apply filters
            if category and metric.category != category:
                continue
            if cost and getattr(metric, 'cost', 'medium') != cost:
                continue
            
            results.append({
                'name': name,
                'category': metric.category,
                'description': metric.description,
                'cost': getattr(metric, 'cost', 'medium'),
                'max_nodes': metric.max_nodes,
            })
    
    return {
        "query": q,
        "filters": {
            "category": category,
            "cost": cost,
        },
        "results": results,
        "count": len(results),
    }