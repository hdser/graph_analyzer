"""
Anomaly Detection Router

API endpoints for anomaly detection functionality.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends

from ..models.requests import (
    AnomalyDetectionRequest,
    ProfileMetricsRequest,
    MetricConfigRequest,
    MetricTransformRequest,
)
from ..models.responses import (
    AlgorithmInfoResponse,
    AnomalyDetectionResponse,
    ProfileMetricsResponse,
    MetricProfileResponse,
    ThresholdInfoResponse,
    GroupAnomalyStatsResponse,
)
from ..services.network_service import NetworkService
from ...engines import (
    AnomalyEngine,
    MetricConfig,
    MetricTransform,
    AlgorithmConfig,
    MetricProfiler,
    NaNStrategy,
    GlobalScaling,
    ScoreNormalization,
    ThresholdMethod,
)


router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])

# Global engine instance
_engine: Optional[AnomalyEngine] = None


def get_engine() -> AnomalyEngine:
    """Get or create anomaly engine instance."""
    global _engine
    if _engine is None:
        _engine = AnomalyEngine(verbose=True)
    return _engine


def get_network_service() -> NetworkService:
    """Get network service instance."""
    from ..services import network_service
    return network_service


@router.get("/algorithms", response_model=Dict[str, AlgorithmInfoResponse])
async def get_algorithms():
    """
    Get available anomaly detection algorithms.
    
    Returns dictionary of algorithm names to their specifications.
    """
    engine = get_engine()
    algorithms = engine.get_available_algorithms()
    
    result = {}
    for name, info in algorithms.items():
        result[name] = AlgorithmInfoResponse(
            name=info.name,
            display_name=info.display_name,
            description=info.description,
            complexity=info.complexity,
            multivariate=info.supports_multivariate,
            requires_sklearn=info.requires_sklearn,
            parameters={
                pname: {
                    "name": spec.name,
                    "type": spec.param_type,
                    "default": spec.default,
                    "min": spec.min_value,
                    "max": spec.max_value,
                    "choices": spec.choices,
                    "description": spec.description,
                }
                for pname, spec in info.parameters.items()
            }
        )
    
    return result


@router.post("/profile", response_model=ProfileMetricsResponse)
async def profile_metrics(request: ProfileMetricsRequest):
    """
    Profile metrics for preprocessing suggestions.
    
    Analyzes specified metrics and returns statistics,
    warnings, and suggested preprocessing configuration.
    """
    service = get_network_service()
    engine = get_engine()
    
    # Get current graph data
    df = service.get_current_metrics_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No graph data loaded")
    
    # Validate metrics exist
    missing = [m for m in request.metrics if m not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Metrics not found: {missing}. Available: {list(df.columns)}"
        )
    
    # Profile metrics
    profiles = engine.profile_metrics(df, request.metrics)
    
    # Get suggested config
    suggested_config = engine.suggest_config(profiles)
    
    # Generate report
    report = engine.generate_profile_report(profiles)
    
    # Convert to response
    profile_responses = {}
    for name, profile in profiles.items():
        profile_responses[name] = MetricProfileResponse(
            name=profile.name,
            dtype=profile.dtype,
            n_samples=profile.n_samples,
            n_unique=profile.n_unique,
            n_missing=profile.n_missing,
            n_zeros=profile.n_zeros,
            n_negative=profile.n_negative,
            n_inf=profile.n_inf,
            min=profile.min_val,
            max=profile.max_val,
            mean=profile.mean,
            median=profile.median,
            std=profile.std,
            skewness=profile.skewness,
            kurtosis=profile.kurtosis,
            p25=profile.p25,
            p75=profile.p75,
            p95=profile.p95,
            p99=profile.p99,
            iqr=profile.iqr,
            suggested_transform=profile.suggested_transform.to_dict(),
            warnings=profile.warnings,
        )
    
    return ProfileMetricsResponse(
        profiles=profile_responses,
        suggested_config=suggested_config.to_dict(),
        report=report,
    )


@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Run anomaly detection on specified metrics.
    
    Supports:
    - Multiple algorithms (zscore, iqr, isolation_forest, lof, dbscan, mahalanobis)
    - Configurable preprocessing
    - Group-aware detection
    - Automatic sampling for large datasets
    """
    service = get_network_service()
    engine = get_engine()
    
    # Get current graph data
    df = service.get_current_metrics_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No graph data loaded")
    
    # Validate metrics exist
    missing = [m for m in request.metrics if m not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Metrics not found: {missing}. Available: {list(df.columns)}"
        )
    
    # Build MetricConfig from request
    config = None
    if request.config:
        per_metric = {}
        for name, transform_req in request.config.per_metric.items():
            per_metric[name] = MetricTransform(
                log=transform_req.log,
                clip_min=transform_req.clip_min,
                clip_max=transform_req.clip_max,
                drop=transform_req.drop,
                weight=transform_req.weight,
                fill_value=transform_req.fill_value,
            )
        
        config = MetricConfig(
            id_column=request.config.id_column,
            group_by=request.config.group_by,
            nan_strategy=NaNStrategy(request.config.nan_strategy),
            per_metric=per_metric,
            global_scaling=GlobalScaling(request.config.global_scaling),
            min_group_size=request.config.min_group_size,
            use_float32=request.config.use_float32,
        )
    
    # Build AlgorithmConfig if provided
    algorithm_config = None
    if request.algorithm_config:
        algorithm_config = AlgorithmConfig(
            algorithm=request.algorithm_config.algorithm,
            parameters=request.algorithm_config.parameters,
            top_n=request.algorithm_config.top_n,
            score_normalization=ScoreNormalization(request.algorithm_config.score_normalization),
            threshold_method=ThresholdMethod(request.algorithm_config.threshold_method),
            threshold_value=request.algorithm_config.threshold_value,
        )
    
    try:
        # Run detection
        result = engine.detect_anomalies(
            df=df,
            metrics=request.metrics,
            algorithm=request.algorithm,
            parameters=request.parameters,
            config=config,
            algorithm_config=algorithm_config,
            sample_size=request.sample_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    # Apply to graph if requested
    node_updates = None
    if request.apply_to_graph:
        node_updates = []
        for node_id, score in result.scores.items():
            node_updates.append({
                "id": node_id,
                request.name: score,
                f"{request.name}_is_anomaly": result.binary_labels.get(node_id, False),
            })
        service.update_node_data(node_updates)
    
    # Build response
    threshold_info = ThresholdInfoResponse(
        method=result.threshold_info.method,
        value=result.threshold_info.value,
        percentile=result.threshold_info.percentile,
        auto_reason=result.threshold_info.auto_reason,
    )
    
    group_results = None
    if result.group_results:
        group_results = {}
        for group_key, stats in result.group_results.items():
            group_results[str(group_key)] = GroupAnomalyStatsResponse(
                group_value=stats.group_value,
                n_samples=stats.n_samples,
                n_anomalies=stats.n_anomalies,
                anomaly_rate=stats.anomaly_rate,
                mean_score=stats.mean_score,
                std_score=stats.std_score,
                threshold_used=stats.threshold_used,
                top_anomalies=stats.top_anomalies,
            )
    
    preprocessing_stats = None
    if result.preprocessing_stats:
        preprocessing_stats = {
            name: stats.to_dict()
            for name, stats in result.preprocessing_stats.items()
        }
    
    return AnomalyDetectionResponse(
        metric_name=request.name,
        algorithm=result.algorithm,
        parameters=result.parameters,
        metrics_used=result.metrics_used,
        threshold_info=threshold_info,
        n_anomalies=result.n_anomalies,
        n_total=result.n_total,
        anomaly_percentage=result.anomaly_rate,
        computation_time=result.computation_time,
        statistics=result.statistics,
        top_anomalies=result.top_anomalies,
        group_results=group_results,
        preprocessing_stats=preprocessing_stats,
        node_updates=node_updates,
    )


@router.get("/presets")
async def get_presets():
    """
    Get available detection presets.
    
    Presets are pre-configured algorithm + config combinations
    for common use cases.
    """
    from engines.anomaly_config import ANOMALY_PRESETS, list_presets
    
    return {
        "presets": list_presets(),
        "details": {
            name: {
                "description": preset["description"],
                "algorithm": preset["algorithm_config"].algorithm,
                "parameters": preset["algorithm_config"].parameters,
            }
            for name, preset in ANOMALY_PRESETS.items()
        }
    }


@router.get("/recommend")
async def recommend_algorithm(
    n_nodes: int,
    n_metrics: int,
    time_constraint: Optional[str] = None,
    memory_constraint: Optional[str] = None,
):
    """
    Get algorithm recommendation based on data characteristics.
    
    Args:
        n_nodes: Number of nodes in graph
        n_metrics: Number of metrics to analyze
        time_constraint: "fast", "moderate", or "slow"
        memory_constraint: "low", "moderate", or "high"
    """
    engine = get_engine()
    
    recommendation = engine.recommend_algorithm(
        n_nodes=n_nodes,
        n_metrics=n_metrics,
        time_constraint=time_constraint,
        memory_constraint=memory_constraint,
    )
    
    # Get info about recommended algorithm
    algorithms = engine.get_available_algorithms()
    info = algorithms.get(recommendation)
    
    return {
        "recommended": recommendation,
        "reason": _get_recommendation_reason(n_nodes, n_metrics, time_constraint, memory_constraint),
        "algorithm_info": info.to_dict() if info else None,
    }


def _get_recommendation_reason(
    n_nodes: int,
    n_metrics: int,
    time_constraint: Optional[str],
    memory_constraint: Optional[str],
) -> str:
    """Generate explanation for recommendation."""
    reasons = []
    
    if time_constraint == "fast":
        reasons.append("fast time constraint requires O(n) algorithm")
    
    if memory_constraint == "low":
        reasons.append("low memory constraint favors simple algorithms")
    
    if n_nodes > 100000:
        reasons.append(f"large graph ({n_nodes:,} nodes) requires efficient algorithm")
    elif n_nodes > 50000:
        reasons.append(f"moderately large graph ({n_nodes:,} nodes)")
    
    if n_metrics == 1:
        reasons.append("single metric benefits from univariate methods")
    elif n_metrics >= 3:
        reasons.append(f"multiple metrics ({n_metrics}) may have correlations")
    
    return "; ".join(reasons) if reasons else "default recommendation for data characteristics"