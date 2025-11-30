"""
Anomaly Detection Router

API endpoints for anomaly detection functionality.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

import numpy as np

from ..models.requests import (
    AnomalyDetectionRequest,
    ProfileMetricsRequest,
    MetricConfigRequest,
    MetricTransformRequest,
    PCARequest,
)
from ..models.responses import (
    AlgorithmInfoResponse,
    AnomalyDetectionResponse,
    ProfileMetricsResponse,
    MetricProfileResponse,
    ThresholdInfoResponse,
    GroupAnomalyStatsResponse,
    AnomalyVisualizationData,
    PCAResponse,
)
from ..services.network_service import NetworkService
from engines import (
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
    from backend.services import network_service
    return network_service


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to a JSON-compatible float."""
    if value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _build_visualization_data(
    result: Any,
    df: Any,
    metrics: List[str],
    algorithm: str,
    parameters: Dict[str, Any]
) -> AnomalyVisualizationData:
    """
    Build additional visualization data for anomaly detection results.
    
    Args:
        result: AnomalyResult from detection
        df: Original DataFrame
        metrics: List of metrics used
        algorithm: Algorithm name
        parameters: Algorithm parameters
    
    Returns:
        AnomalyVisualizationData with enhanced visualization info
    """
    # Basic threshold info
    threshold_value = _safe_float(result.threshold_info.value, 0.5)
    scores = np.array(list(result.scores.values()))
    
    scores_above = int(np.sum(scores >= threshold_value))
    scores_below = int(np.sum(scores < threshold_value))
    
    # Build histogram data
    n_bins = 30
    hist_counts, bin_edges = np.histogram(scores, bins=n_bins, range=(0, 1))
    
    # Count anomalies in each bin
    anomaly_scores = np.array([
        result.scores[node_id] 
        for node_id, is_anomaly in result.binary_labels.items() 
        if is_anomaly
    ])
    anomaly_counts, _ = np.histogram(anomaly_scores, bins=bin_edges)
    
    # Per-metric statistics for multivariate
    per_metric_stats = None
    per_metric_contributions = None
    
    if len(metrics) > 1:
        per_metric_stats = {}
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].values
                values = np.nan_to_num(values, nan=0.0)
                per_metric_stats[metric] = {
                    'mean': _safe_float(np.mean(values)),
                    'std': _safe_float(np.std(values)),
                    'min': _safe_float(np.min(values)),
                    'max': _safe_float(np.max(values)),
                    'median': _safe_float(np.median(values)),
                }
    
    # Algorithm-specific details
    algorithm_details = {
        'algorithm': algorithm,
        'parameters': parameters,
    }
    
    if algorithm == 'isolation_forest':
        algorithm_details['type'] = 'ensemble'
        algorithm_details['n_estimators'] = parameters.get('n_estimators', 100)
        algorithm_details['contamination'] = parameters.get('contamination', 'auto')
    elif algorithm == 'lof':
        algorithm_details['type'] = 'density'
        algorithm_details['n_neighbors'] = parameters.get('n_neighbors', 20)
    elif algorithm == 'dbscan':
        algorithm_details['type'] = 'clustering'
        algorithm_details['eps'] = parameters.get('eps', 0.5)
        algorithm_details['min_samples'] = parameters.get('min_samples', 5)
    elif algorithm == 'mahalanobis':
        algorithm_details['type'] = 'distance'
    elif algorithm in ('zscore', 'iqr'):
        algorithm_details['type'] = 'statistical'
        if algorithm == 'zscore':
            algorithm_details['threshold'] = parameters.get('threshold', 3.0)
        else:
            algorithm_details['k'] = parameters.get('k', 1.5)
    
    return AnomalyVisualizationData(
        threshold_value=threshold_value,
        scores_above_threshold=scores_above,
        scores_below_threshold=scores_below,
        score_bins=[_safe_float(b) for b in bin_edges.tolist()],
        score_counts=[int(c) for c in hist_counts.tolist()],
        anomaly_counts=[int(c) for c in anomaly_counts.tolist()],
        per_metric_stats=per_metric_stats,
        per_metric_contributions=per_metric_contributions,
        algorithm_details=algorithm_details,
    )


# =============================================================================
# Existing Endpoints
# =============================================================================

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
    - Multiple algorithms (zscore, iqr, isolation_forest, lof, dbscan, mahalanobis,
      pca_reconstruction, one_class_svm)
    - Configurable preprocessing
    - Group-aware detection
    - Automatic sampling for large datasets
    - Filtering to specific node IDs
    """
    service = get_network_service()
    engine = get_engine()
    
    # Get current graph data
    df = service.get_current_metrics_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No graph data loaded")
    
    # Filter to specific nodes if requested
    if request.node_ids:
        id_col = 'avatar' if 'avatar' in df.columns else 'id' if 'id' in df.columns else None
        if id_col:
            df = df[df[id_col].astype(str).isin(request.node_ids)]
        else:
            # Try filtering by index
            df = df[df.index.astype(str).isin(request.node_ids)]
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No matching nodes found for the specified node_ids"
            )
    
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
    
    # Build visualization data
    visualization_data = _build_visualization_data(
        result=result,
        df=df,
        metrics=request.metrics,
        algorithm=request.algorithm,
        parameters=request.parameters,
    )
    
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
        visualization_data=visualization_data,
    )


# =============================================================================
# PCA Analysis Endpoint
# =============================================================================

@router.post("/pca", response_model=PCAResponse)
async def run_pca_analysis(request: PCARequest):
    """
    Run PCA (Principal Component Analysis) on specified metrics.
    
    Supports filtering to specific node IDs via request.node_ids.
    
    Returns:
    - Principal component scores for each node
    - Explained variance ratios
    - Feature loadings (contribution of each feature to each component)
    - Reconstruction errors for anomaly detection
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    service = get_network_service()
    
    # Get current graph data
    df = service.get_current_metrics_df()
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No graph data loaded")
    
    # Filter to specific nodes if requested
    if request.node_ids:
        id_col = 'avatar' if 'avatar' in df.columns else 'id' if 'id' in df.columns else None
        if id_col:
            df = df[df[id_col].astype(str).isin(request.node_ids)]
        else:
            df = df[df.index.astype(str).isin(request.node_ids)]
        
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No matching nodes found for the specified node_ids"
            )
    
    # Validate metrics exist
    missing = [m for m in request.metrics if m not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Metrics not found: {missing}. Available: {list(df.columns)}"
        )
    
    if len(request.metrics) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 metrics are required for PCA analysis"
        )
    
    try:
        # Extract data matrix
        X = df[request.metrics].values.astype(np.float64)
        
        # Get node IDs
        if 'avatar' in df.columns:
            node_ids = df['avatar'].astype(str).tolist()
        elif 'id' in df.columns:
            node_ids = df['id'].astype(str).tolist()
        else:
            node_ids = df.index.astype(str).tolist()
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_samples, n_features = X.shape
        
        # Standardize if requested
        if request.standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_scaled = X
        
        # Determine number of components
        max_components = min(n_samples, n_features)
        
        if request.n_components == "auto":
            # Use enough components to explain 95% variance
            n_components = min(0.95, max_components)
        elif request.n_components.startswith("0."):
            # Variance ratio
            n_components = float(request.n_components)
        else:
            # Fixed number
            n_components = min(int(request.n_components), max_components)
        
        # Fit PCA
        pca = PCA(n_components=n_components, random_state=42)
        X_transformed = pca.fit_transform(X_scaled)
        
        # Get actual number of components
        actual_n_components = pca.n_components_
        
        # Calculate reconstruction error
        X_reconstructed = pca.inverse_transform(X_transformed)
        reconstruction_errors = np.sqrt(np.sum((X_scaled - X_reconstructed) ** 2, axis=1))
        reconstruction_errors = np.nan_to_num(reconstruction_errors, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Build loadings dict (feature contributions to each component)
        loadings = {}
        for i in range(actual_n_components):
            loadings[f"PC{i+1}"] = [_safe_float(v) for v in pca.components_[i]]
        
        # Build transformed data dict
        transformed_data = {}
        for i in range(actual_n_components):
            transformed_data[f"PC{i+1}"] = [_safe_float(v) for v in X_transformed[:, i]]
        
        # Calculate total variance explained
        total_variance = float(np.sum(pca.explained_variance_ratio_))
        
        return PCAResponse(
            n_components=actual_n_components,
            n_samples=n_samples,
            features=request.metrics,
            explained_variance_ratio=[_safe_float(v) for v in pca.explained_variance_ratio_],
            total_variance_explained=_safe_float(total_variance),
            loadings=loadings,
            transformed_data=transformed_data,
            node_ids=node_ids,
            reconstruction_errors=[_safe_float(v) for v in reconstruction_errors],
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PCA analysis failed: {str(e)}")


# =============================================================================
# Other Endpoints
# =============================================================================

@router.get("/presets")
async def get_presets():
    """
    Get available detection presets.
    
    Presets are pre-configured algorithm + config combinations
    for common use cases.
    """
    try:
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
    except ImportError:
        return {"presets": [], "details": {}}


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