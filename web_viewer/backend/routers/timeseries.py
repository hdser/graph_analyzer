"""
Timeseries Router

API endpoints for timeseries analysis including:
- Metric timeseries (aggregated across nodes)
- Network summary timeseries
- Node trajectories
- Trend detection
- Distribution comparisons
- Cohort analysis
- Batch metrics loading for optimized timeline scrubbing
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..config import settings

from ..models.timeseries import (
    AggregationType,
    TimeseriesData,
    NetworkTimeseriesData,
    NodeTrajectoriesResponse,
    TrendAnalysis,
    DistributionComparison,
    CohortDefinition,
    CohortTrajectory,
    NodeTrajectoriesRequest,
    DistributionComparisonRequest,
    CohortAnalysisRequest,
)
from engines.timeseries_engine import TimeseriesEngine, timeseries_engine


router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])


def get_engine() -> TimeseriesEngine:
    """Get timeseries engine instance."""
    return timeseries_engine


# =============================================================================
# Request Models (for POST endpoints)
# =============================================================================

class MetricTimeseriesRequest(BaseModel):
    """Request for metric timeseries."""
    metric: str
    aggregation: AggregationType = AggregationType.MEAN
    start_block: Optional[int] = None
    end_block: Optional[int] = None
    include_trend: bool = True


# =============================================================================
# Network-Level Timeseries Endpoints
# =============================================================================

@router.get("/{base_sql_file}/metrics/{metric}", response_model=TimeseriesData)
async def get_metric_timeseries(
    base_sql_file: str,
    metric: str,
    aggregation: AggregationType = Query(AggregationType.MEAN, description="Aggregation method"),
    start_block: Optional[int] = Query(None, description="Start block number"),
    end_block: Optional[int] = Query(None, description="End block number"),
    include_trend: bool = Query(True, description="Include trend analysis"),
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get timeseries of a metric aggregated across all nodes.
    
    Returns the metric value (aggregated using the specified method) for each
    snapshot, along with statistics and optional trend analysis.
    
    Args:
        base_sql_file: Base SQL file name
        metric: Metric name to track (e.g., 'pagerank', 'in_degree')
        aggregation: How to aggregate across nodes (mean, median, sum, etc.)
        start_block: Optional start block filter
        end_block: Optional end block filter
        include_trend: Whether to compute trend analysis
    """
    try:
        result = await asyncio.to_thread(
            engine.get_metric_timeseries,
            base_sql_file,
            metric,
            aggregation,
            start_block,
            end_block,
            include_trend
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeseries computation failed: {str(e)}")


@router.get("/{base_sql_file}/network-summary", response_model=NetworkTimeseriesData)
async def get_network_summary_timeseries(
    base_sql_file: str,
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get network-level statistics over time.
    
    Returns node count, edge count, density, and growth rates
    for each snapshot.
    """
    try:
        result = await asyncio.to_thread(
            engine.get_network_summary_timeseries,
            base_sql_file
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Network summary failed: {str(e)}")


# =============================================================================
# Node-Level Trajectory Endpoints
# =============================================================================

@router.post("/{base_sql_file}/node-trajectories", response_model=NodeTrajectoriesResponse)
async def get_node_trajectories(
    base_sql_file: str,
    request: NodeTrajectoriesRequest,
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get metric trajectories for specific nodes over time.
    
    Tracks the value of a metric for each specified node across all snapshots.
    Useful for understanding how individual nodes evolve.
    
    Args:
        base_sql_file: Base SQL file name
        request: Contains node_ids, metric, and options
    """
    if len(request.node_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 nodes per request"
        )
    
    try:
        result = await asyncio.to_thread(
            engine.get_node_trajectories,
            base_sql_file,
            request.node_ids,
            request.metric,
            request.include_statistics,
            request.include_trend
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trajectory computation failed: {str(e)}")


@router.get("/{base_sql_file}/node/{node_id}/trajectory")
async def get_single_node_trajectory(
    base_sql_file: str,
    node_id: str,
    metric: str = Query(..., description="Metric to track"),
    include_trend: bool = Query(False, description="Include trend analysis"),
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get trajectory for a single node.
    
    Convenience endpoint for tracking one node's metric over time.
    """
    try:
        result = await asyncio.to_thread(
            engine.get_node_trajectories,
            base_sql_file,
            [node_id],
            metric,
            True,
            include_trend
        )
        
        if node_id not in result.trajectories:
            raise HTTPException(
                status_code=404,
                detail=f"Node {node_id} not found in any snapshot"
            )
        
        return result.trajectories[node_id]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Trend Analysis Endpoints
# =============================================================================

@router.get("/{base_sql_file}/trends/{metric}", response_model=TrendAnalysis)
async def get_metric_trend(
    base_sql_file: str,
    metric: str,
    aggregation: AggregationType = Query(AggregationType.MEAN),
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get trend analysis for a metric.
    
    Computes linear regression, direction, significance, and volatility
    metrics for the timeseries.
    """
    try:
        result = await asyncio.to_thread(
            engine.detect_metric_trend,
            base_sql_file,
            metric,
            aggregation
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


# =============================================================================
# Distribution Comparison Endpoints
# =============================================================================

@router.get("/{base_sql_file}/compare/{metric}", response_model=DistributionComparison)
async def compare_distributions(
    base_sql_file: str,
    metric: str,
    from_block: int = Query(..., description="Earlier snapshot block number"),
    to_block: int = Query(..., description="Later snapshot block number"),
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Compare metric distributions between two snapshots.
    
    Uses Kolmogorov-Smirnov test and computes shift metrics to understand
    how the distribution of a metric changed between two points in time.
    """
    if from_block >= to_block:
        raise HTTPException(
            status_code=400,
            detail="from_block must be less than to_block"
        )
    
    try:
        result = await asyncio.to_thread(
            engine.compare_distributions,
            base_sql_file,
            metric,
            from_block,
            to_block
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.post("/{base_sql_file}/compare", response_model=DistributionComparison)
async def compare_distributions_post(
    base_sql_file: str,
    request: DistributionComparisonRequest,
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Compare metric distributions (POST version with body).
    """
    return await compare_distributions(
        base_sql_file,
        request.metric,
        request.from_block,
        request.to_block,
        engine
    )


# =============================================================================
# Cohort Analysis Endpoints
# =============================================================================

@router.post("/{base_sql_file}/cohort", response_model=CohortTrajectory)
async def get_cohort_trajectory(
    base_sql_file: str,
    cohort_definition: CohortDefinition,
    metric: str = Query(..., description="Metric to track"),
    aggregation: AggregationType = Query(AggregationType.MEAN),
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Track a cohort of nodes over time.
    
    A cohort is a group of nodes that share a common characteristic,
    such as appearing in the same snapshot or having similar initial values.
    
    Tracks:
    - How many cohort members remain active
    - The aggregated metric value for the cohort
    - Retention rate
    """
    try:
        result = await asyncio.to_thread(
            engine.get_cohort_trajectory,
            base_sql_file,
            cohort_definition,
            metric,
            aggregation
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohort analysis failed: {str(e)}")


@router.post("/{base_sql_file}/cohorts/compare")
async def compare_cohorts(
    base_sql_file: str,
    request: CohortAnalysisRequest,
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Compare multiple cohorts over time.
    
    Analyzes each cohort and provides comparative statistics.
    """
    if len(request.cohort_definitions) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 cohorts per comparison"
        )
    
    try:
        cohort_results = []
        
        for cohort_def in request.cohort_definitions:
            result = await asyncio.to_thread(
                engine.get_cohort_trajectory,
                base_sql_file,
                cohort_def,
                request.metric,
                request.aggregation
            )
            cohort_results.append(result)
        
        # Find best/worst performers
        best_performer = None
        worst_performer = None
        highest_retention = None
        
        if cohort_results:
            # By final metric value
            with_values = [c for c in cohort_results if c.data_points and c.data_points[-1].metric_mean is not None]
            if with_values:
                sorted_by_value = sorted(with_values, key=lambda c: c.data_points[-1].metric_mean or 0, reverse=True)
                best_performer = sorted_by_value[0].cohort_definition.name
                worst_performer = sorted_by_value[-1].cohort_definition.name
            
            # By retention
            sorted_by_retention = sorted(cohort_results, key=lambda c: c.retention_rate, reverse=True)
            highest_retention = sorted_by_retention[0].cohort_definition.name
        
        return {
            "base_sql_file": base_sql_file,
            "metric": request.metric,
            "cohorts": [c.model_dump() for c in cohort_results],
            "best_performing_cohort": best_performer,
            "worst_performing_cohort": worst_performer,
            "highest_retention_cohort": highest_retention
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohort comparison failed: {str(e)}")


# =============================================================================
# Batch Loading Endpoints (Optimized for Timeline)
# =============================================================================

class BatchMetricsRequest(BaseModel):
    """Request for batch metrics loading."""
    metrics: List[str] = Field(..., description="List of metric names to fetch")
    aggregation: AggregationType = Field(
        default=AggregationType.MEAN,
        description="Aggregation method for all metrics"
    )
    include_trend: bool = Field(default=False, description="Include trend analysis")


@router.post("/{base_sql_file}/metrics/batch")
async def get_metrics_timeseries_batch(
    base_sql_file: str,
    request: BatchMetricsRequest,
    engine: TimeseriesEngine = Depends(get_engine)
) -> Dict[str, Any]:
    """
    Load multiple metrics across all snapshots in one request.

    This is much more efficient than making individual requests when
    displaying timeline charts with multiple metrics.

    Returns:
        {
            "base_sql_file": "...",
            "metrics": {
                "metric1": {...timeseries data...},
                "metric2": {...timeseries data...},
                ...
            },
            "snapshot_count": N,
            "success": ["metric1", "metric2"],
            "failed": []
        }
    """
    if len(request.metrics) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 metrics per batch request"
        )

    results: Dict[str, Any] = {}
    success = []
    failed = []

    # Use ThreadPoolExecutor for parallel loading
    max_workers = min(settings.TIMESERIES_BATCH_SIZE, len(request.metrics))

    async def fetch_metric(metric: str) -> tuple:
        """Fetch a single metric's timeseries."""
        try:
            result = await asyncio.to_thread(
                engine.get_metric_timeseries,
                base_sql_file,
                metric,
                request.aggregation,
                None,  # start_block
                None,  # end_block
                request.include_trend
            )
            return metric, result.model_dump(), None
        except Exception as e:
            return metric, None, str(e)

    # Run all metric fetches concurrently
    tasks = [fetch_metric(metric) for metric in request.metrics]
    completed = await asyncio.gather(*tasks)

    snapshot_count = 0

    for metric, data, error in completed:
        if data:
            results[metric] = data
            success.append(metric)
            if 'data_points' in data:
                snapshot_count = max(snapshot_count, len(data['data_points']))
        else:
            results[metric] = {"error": error}
            failed.append(metric)

    return {
        "base_sql_file": base_sql_file,
        "metrics": results,
        "snapshot_count": snapshot_count,
        "success": success,
        "failed": failed,
        "total_requested": len(request.metrics)
    }


@router.get("/{base_sql_file}/snapshots/list")
async def list_snapshots_for_timeseries(
    base_sql_file: str,
    engine: TimeseriesEngine = Depends(get_engine)
) -> Dict[str, Any]:
    """
    Get list of snapshots available for timeseries analysis.

    Returns basic metadata about each snapshot for timeline display.
    """
    from ..services.snapshot_analysis_service import snapshot_analysis_service

    try:
        snapshots = await asyncio.to_thread(
            snapshot_analysis_service.list_analyzed_snapshots,
            base_sql_file
        )

        return {
            "base_sql_file": base_sql_file,
            "snapshot_count": len(snapshots),
            "snapshots": [
                {
                    "snapshot_id": s.snapshot_id,
                    "block_number": s.block_number,
                    "timestamp": s.analysis_timestamp,
                    "node_count": s.node_count,
                    "metrics_count": len(s.metrics_computed)
                }
                for s in snapshots
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Utility Endpoints
# =============================================================================

@router.get("/{base_sql_file}/available-metrics")
async def get_available_metrics(
    base_sql_file: str,
    engine: TimeseriesEngine = Depends(get_engine)
):
    """
    Get list of metrics available for timeseries analysis.
    
    Returns metrics that have been computed for at least one snapshot.
    """
    from ..services.snapshot_analysis_service import snapshot_analysis_service
    
    try:
        # Get analyzed snapshots
        analyzed = await asyncio.to_thread(
            snapshot_analysis_service.list_analyzed_snapshots,
            base_sql_file
        )
        
        # Collect all metrics
        all_metrics = set()
        for snapshot in analyzed:
            all_metrics.update(snapshot.metrics_computed)
        
        return {
            "base_sql_file": base_sql_file,
            "metrics": sorted(list(all_metrics)),
            "snapshots_with_analysis": len(analyzed)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{base_sql_file}/aggregations")
async def get_available_aggregations():
    """
    Get list of available aggregation methods.
    """
    return {
        "aggregations": [
            {"value": agg.value, "name": agg.name, "description": _get_agg_description(agg)}
            for agg in AggregationType
        ]
    }


def _get_agg_description(agg: AggregationType) -> str:
    """Get description for an aggregation type."""
    descriptions = {
        AggregationType.MEAN: "Average value across all nodes",
        AggregationType.MEDIAN: "Median value (50th percentile)",
        AggregationType.SUM: "Sum of all values",
        AggregationType.MIN: "Minimum value",
        AggregationType.MAX: "Maximum value",
        AggregationType.STD: "Standard deviation",
        AggregationType.COUNT: "Number of nodes with valid values",
        AggregationType.P10: "10th percentile",
        AggregationType.P25: "25th percentile",
        AggregationType.P75: "75th percentile",
        AggregationType.P90: "90th percentile",
    }
    return descriptions.get(agg, "")