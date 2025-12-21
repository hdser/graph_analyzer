"""
Metric Resolver

Resolves metric requests to a list of metrics to compute.
Handles presets, categories, individual metrics, and exclusions.
"""

import logging
from typing import List, Optional, Set, Dict, Any

from .registry import (
    METRIC_REGISTRY,
    METRIC_CATEGORIES,
    METRIC_PRESETS,
    MetricDefinition,
    get_metric,
    get_category_metrics,
    get_preset_metrics,
)

logger = logging.getLogger(__name__)


class MetricResolver:
    """
    Resolves metric computation requests to a concrete list of metrics.
    
    Handles:
    - Preset expansion (e.g., "essential" -> list of metrics)
    - Category expansion (e.g., "centrality" -> all centrality metrics)
    - Individual metric selection
    - Metric exclusion
    - Dependency resolution
    - Cost-based filtering
    - Size-based filtering
    """
    
    def __init__(self):
        self.registry = METRIC_REGISTRY
        self.categories = METRIC_CATEGORIES
        self.presets = METRIC_PRESETS
    
    def resolve(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        exclude_metrics: Optional[List[str]] = None,
        skip_expensive: bool = False,
        max_nodes: Optional[int] = None,
        require_connected: Optional[bool] = None,
    ) -> List[MetricDefinition]:
        """
        Resolve a metric request to a list of MetricDefinitions.
        
        Resolution Priority:
        1. If `metrics` provided -> Use exact metric list
        2. If `categories` provided -> Expand to metrics in those categories
        3. If `preset` provided -> Use preset definition
        4. Else -> Default to "basic" preset
        
        Args:
            preset: Preset name (basic, essential, moderate, comprehensive, all)
            categories: List of category names
            metrics: List of individual metric names
            exclude_metrics: Metrics to exclude from final list
            skip_expensive: If True, skip metrics with cost='very_high'
            max_nodes: Skip metrics that require fewer nodes than this
            require_connected: If False, skip metrics requiring connected graph
            
        Returns:
            List of MetricDefinition objects to compute
        """
        resolved_names: Set[str] = set()
        
        # Step 1: Collect metrics based on request type
        if metrics:
            # Explicit metric list provided
            resolved_names = self._resolve_explicit_metrics(metrics)
            logger.debug(f"Resolved explicit metrics: {resolved_names}")
            
        elif categories:
            # Category-based selection
            resolved_names = self._resolve_categories(categories)
            logger.debug(f"Resolved categories {categories}: {len(resolved_names)} metrics")
            
        elif preset:
            # Preset-based selection
            resolved_names = self._resolve_preset(preset)
            logger.debug(f"Resolved preset '{preset}': {len(resolved_names)} metrics")
            
        else:
            # Default to basic preset
            resolved_names = self._resolve_preset("basic")
            logger.debug(f"Using default 'basic' preset: {len(resolved_names)} metrics")
        
        # Step 2: Apply exclusions
        if exclude_metrics:
            before = len(resolved_names)
            resolved_names -= set(exclude_metrics)
            logger.debug(f"Excluded {before - len(resolved_names)} metrics")
        
        # Step 3: Resolve dependencies
        resolved_names = self._resolve_dependencies(resolved_names)
        
        # Step 4: Convert to MetricDefinition objects
        definitions = [
            self.registry[name] for name in resolved_names
            if name in self.registry
        ]
        
        # Step 5: Apply filters
        definitions = self._apply_filters(
            definitions,
            skip_expensive=skip_expensive,
            max_nodes=max_nodes,
            require_connected=require_connected,
        )
        
        # Step 6: Sort by cost for optimal computation order
        definitions = self._sort_by_cost(definitions)
        
        logger.info(f"Resolved {len(definitions)} metrics to compute")
        return definitions
    
    def _resolve_explicit_metrics(self, metrics: List[str]) -> Set[str]:
        """Resolve explicit metric names, validating they exist."""
        resolved = set()
        for name in metrics:
            if name in self.registry:
                resolved.add(name)
            else:
                # Check if it's a category name (common mistake)
                if name in self.categories:
                    logger.warning(f"'{name}' is a category, not a metric. Use categories=['{name}'] instead.")
                    resolved.update(get_category_metrics(name))
                # Check if it's a preset name
                elif name in self.presets:
                    logger.warning(f"'{name}' is a preset, not a metric. Use preset='{name}' instead.")
                    resolved.update(get_preset_metrics(name))
                else:
                    logger.warning(f"Unknown metric: '{name}'. Skipping.")
        return resolved
    
    def _resolve_categories(self, categories: List[str]) -> Set[str]:
        """Resolve category names to metrics."""
        resolved = set()
        for category in categories:
            if category in self.categories:
                resolved.update(get_category_metrics(category))
            else:
                logger.warning(f"Unknown category: '{category}'. Skipping.")
        return resolved
    
    def _resolve_preset(self, preset: str) -> Set[str]:
        """Resolve preset name to metrics."""
        if preset not in self.presets:
            logger.warning(f"Unknown preset: '{preset}'. Using 'basic'.")
            preset = "basic"
        
        return set(get_preset_metrics(preset))
    
    def _resolve_dependencies(self, metrics: Set[str]) -> Set[str]:
        """Add any metrics that are dependencies of requested metrics."""
        resolved = set(metrics)
        added = True
        
        while added:
            added = False
            for name in list(resolved):
                if name in self.registry:
                    deps = self.registry[name].dependencies
                    for dep in deps:
                        if dep not in resolved and dep in self.registry:
                            resolved.add(dep)
                            added = True
                            logger.debug(f"Added dependency '{dep}' for '{name}'")
        
        return resolved
    
    def _apply_filters(
        self,
        definitions: List[MetricDefinition],
        skip_expensive: bool = False,
        max_nodes: Optional[int] = None,
        require_connected: Optional[bool] = None,
    ) -> List[MetricDefinition]:
        """Apply filters to remove metrics that can't or shouldn't be computed."""
        filtered = []
        
        for defn in definitions:
            # Skip expensive metrics if requested
            if skip_expensive and defn.cost == "very_high":
                logger.debug(f"Skipping expensive metric: {defn.name}")
                continue
            
            # Skip if graph is too large
            if max_nodes is not None and defn.max_nodes is not None:
                if max_nodes > defn.max_nodes:
                    logger.debug(f"Skipping {defn.name}: max_nodes={defn.max_nodes}, graph has {max_nodes}")
                    continue
            
            # Skip if requires connected but graph isn't
            if require_connected is False and defn.requires_connected:
                logger.debug(f"Skipping {defn.name}: requires connected graph")
                continue
            
            filtered.append(defn)
        
        return filtered
    
    def _sort_by_cost(self, definitions: List[MetricDefinition]) -> List[MetricDefinition]:
        """Sort metrics by computational cost (cheapest first)."""
        cost_order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
        return sorted(definitions, key=lambda d: cost_order.get(d.cost, 2))
    
    def get_metrics_info(
        self,
        preset: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get information about what metrics would be computed.
        Useful for preview/validation before running.
        """
        definitions = self.resolve(
            preset=preset,
            categories=categories,
            metrics=metrics,
        )
        
        by_category = {}
        for defn in definitions:
            if defn.category not in by_category:
                by_category[defn.category] = []
            by_category[defn.category].append(defn.name)
        
        return {
            "total_metrics": len(definitions),
            "by_category": by_category,
            "metrics": [
                {
                    "name": d.name,
                    "category": d.category,
                    "cost": d.cost,
                    "output_columns": d.output_columns,
                }
                for d in definitions
            ],
            "estimated_cost": self._estimate_cost(definitions),
        }
    
    def _estimate_cost(self, definitions: List[MetricDefinition]) -> str:
        """Estimate overall computation cost."""
        costs = [d.cost for d in definitions]
        
        if "very_high" in costs:
            return "very_high"
        elif costs.count("high") >= 3:
            return "high"
        elif "high" in costs or costs.count("medium") >= 5:
            return "medium"
        else:
            return "low"


# Module-level resolver instance
_resolver = MetricResolver()


def resolve_metrics(
    preset: Optional[str] = None,
    categories: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    exclude_metrics: Optional[List[str]] = None,
    skip_expensive: bool = False,
    max_nodes: Optional[int] = None,
    require_connected: Optional[bool] = None,
) -> List[MetricDefinition]:
    """Convenience function to resolve metrics without instantiating resolver."""
    return _resolver.resolve(
        preset=preset,
        categories=categories,
        metrics=metrics,
        exclude_metrics=exclude_metrics,
        skip_expensive=skip_expensive,
        max_nodes=max_nodes,
        require_connected=require_connected,
    )