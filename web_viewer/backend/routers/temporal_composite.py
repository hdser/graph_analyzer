"""
Temporal Composite Router

API endpoints for temporal composite metrics including:
- Computing temporal metrics (velocity, stability, momentum, etc.)
- Previewing temporal metrics
- Applying presets
- Listing available operations
"""

import asyncio
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..models.temporal_composite import (
    TemporalOperation,
    TemporalCompositeConfig,
    TemporalCompositeResult,
    TemporalPreviewResult,
    TemporalPresetInfo,
    TemporalPresetsResponse,
    AvailableOperationsResponse,
    TemporalPresetRequest,
)
from engines.temporal_composite_engine import (
    TemporalCompositeEngine,
    temporal_composite_engine
)


router = APIRouter(prefix="/api/temporal", tags=["temporal-composite"])


def get_engine() -> TemporalCompositeEngine:
    """Get temporal composite engine instance."""
    return temporal_composite_engine


# =============================================================================
# Computation Endpoints
# =============================================================================

@router.post("/compute", response_model=TemporalCompositeResult)
async def compute_temporal_composite(
    config: TemporalCompositeConfig,
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Compute a temporal composite metric.
    
    Creates a new metric by applying a temporal operation (velocity, stability,
    momentum, etc.) to a base metric across historical snapshots.
    
    Args:
        config: Full configuration for the temporal composite
        
    Returns:
        TemporalCompositeResult with computed values and statistics
    """
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation failed: {str(e)}")


@router.post("/preview")
async def preview_temporal_composite(
    config: TemporalCompositeConfig,
    sample_size: int = Query(10, ge=1, le=50, description="Number of top/bottom nodes to show"),
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Preview a temporal composite metric without saving.
    
    Returns statistics, histogram, and sample values to help evaluate
    the metric before creating it.
    """
    # Temporarily disable saving for preview
    config_copy = config.model_copy()
    config_copy.save = False
    
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config_copy
        )
        
        # Build preview response
        values = result.values or {}
        
        # Sort for top/bottom
        sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=True)
        top_nodes = [{"node_id": k, "value": v} for k, v in sorted_items[:sample_size]]
        bottom_nodes = [{"node_id": k, "value": v} for k, v in sorted_items[-sample_size:]]
        
        # Compute histogram
        import numpy as np
        arr = np.array(list(values.values()))
        arr = arr[np.isfinite(arr)]
        
        if len(arr) > 0:
            hist_counts, bin_edges = np.histogram(arr, bins=30)
            histogram_bins = [float(b) for b in bin_edges]
            histogram_counts = [int(c) for c in hist_counts]
        else:
            histogram_bins = []
            histogram_counts = []
        
        # Compute correlation with base metric
        from ..services.snapshot_analysis_service import snapshot_analysis_service
        base_result = await asyncio.to_thread(
            snapshot_analysis_service.get_metric_values,
            config.base_sql_file,
            config.target_block,
            config.base_metric,
            True
        )
        
        correlation = 0.0
        if base_result and base_result.values:
            base_values = base_result.values
            common_nodes = set(values.keys()) & set(base_values.keys())
            if len(common_nodes) > 2:
                temporal_arr = np.array([values[n] for n in common_nodes])
                base_arr = np.array([base_values[n] for n in common_nodes])
                valid = np.isfinite(temporal_arr) & np.isfinite(base_arr)
                if valid.sum() > 2:
                    correlation = float(np.corrcoef(temporal_arr[valid], base_arr[valid])[0, 1])
        
        return TemporalPreviewResult(
            name=config.name,
            formula_description=result.formula_description,
            statistics=result.statistics,
            histogram_bins=histogram_bins,
            histogram_counts=histogram_counts,
            top_nodes=top_nodes,
            bottom_nodes=bottom_nodes,
            correlation_with_base=correlation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


# =============================================================================
# Preset Endpoints
# =============================================================================

@router.get("/presets", response_model=TemporalPresetsResponse)
async def get_presets(
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Get list of available preset temporal metrics.
    
    Presets are pre-configured temporal metrics for common use cases.
    """
    presets = engine.get_presets()
    categories = list(set(p.category for p in presets))
    
    return TemporalPresetsResponse(
        presets=presets,
        categories=sorted(categories)
    )


@router.post("/preset/{preset_id}", response_model=TemporalCompositeResult)
async def apply_preset(
    preset_id: str,
    request: TemporalPresetRequest,
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Apply a preset temporal metric.
    
    Convenience endpoint for applying pre-configured temporal metrics.
    
    Args:
        preset_id: ID of the preset to apply
        request: Target snapshot and optional overrides
    """
    try:
        result = await asyncio.to_thread(
            engine.apply_preset,
            preset_id,
            request.base_sql_file,
            request.target_block,
            request.window_blocks,
            request.save
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preset application failed: {str(e)}")


# =============================================================================
# Operation Information Endpoints
# =============================================================================

@router.get("/operations", response_model=AvailableOperationsResponse)
async def get_available_operations(
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Get information about all available temporal operations.
    
    Returns details about each operation including formula, requirements,
    and interpretation guidance.
    """
    return engine.get_available_operations()


@router.get("/operations/{operation}")
async def get_operation_info(
    operation: TemporalOperation,
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Get detailed information about a specific operation.
    """
    ops = engine.get_available_operations()
    
    for op_info in ops.operations:
        if op_info.operation == operation:
            return op_info
    
    raise HTTPException(status_code=404, detail=f"Operation not found: {operation}")


# =============================================================================
# Convenience Endpoints for Common Operations
# =============================================================================

@router.get("/{base_sql_file}/velocity/{metric}")
async def compute_velocity(
    base_sql_file: str,
    metric: str,
    target_block: int = Query(..., description="Target block number"),
    window: int = Query(5, ge=2, le=20, description="Number of snapshots"),
    normalize: bool = Query(True, description="Normalize to [0,1]"),
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Compute velocity (rate of change) for a metric.
    
    Convenience endpoint for the most common temporal operation.
    """
    from ..models.temporal_composite import TemporalOperationConfig
    
    config = TemporalCompositeConfig(
        name=f"{metric}_velocity",
        base_metric=metric,
        temporal_config=TemporalOperationConfig(
            operation=TemporalOperation.VELOCITY,
            window_blocks=window,
            normalize_output=normalize
        ),
        base_sql_file=base_sql_file,
        target_block=target_block,
        save=False
    )
    
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{base_sql_file}/stability/{metric}")
async def compute_stability(
    base_sql_file: str,
    metric: str,
    target_block: int = Query(..., description="Target block number"),
    window: int = Query(5, ge=3, le=20, description="Number of snapshots"),
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Compute stability (consistency over time) for a metric.
    
    Returns values in [0, 1] where 1 = perfectly stable.
    """
    from ..models.temporal_composite import TemporalOperationConfig
    
    config = TemporalCompositeConfig(
        name=f"{metric}_stability",
        base_metric=metric,
        temporal_config=TemporalOperationConfig(
            operation=TemporalOperation.STABILITY,
            window_blocks=window,
            normalize_output=False  # Already [0,1]
        ),
        base_sql_file=base_sql_file,
        target_block=target_block,
        save=False
    )
    
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{base_sql_file}/momentum/{metric}")
async def compute_momentum(
    base_sql_file: str,
    metric: str,
    target_block: int = Query(..., description="Target block number"),
    window: int = Query(5, ge=3, le=20, description="Number of snapshots"),
    decay: float = Query(0.9, ge=0.5, le=0.99, description="Decay factor"),
    normalize: bool = Query(True, description="Normalize to [0,1]"),
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Compute momentum (exponentially weighted trend) for a metric.
    """
    from ..models.temporal_composite import TemporalOperationConfig
    
    config = TemporalCompositeConfig(
        name=f"{metric}_momentum",
        base_metric=metric,
        temporal_config=TemporalOperationConfig(
            operation=TemporalOperation.MOMENTUM,
            window_blocks=window,
            decay_factor=decay,
            normalize_output=normalize
        ),
        base_sql_file=base_sql_file,
        target_block=target_block,
        save=False
    )
    
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{base_sql_file}/age-weighted/{metric}")
async def compute_age_weighted(
    base_sql_file: str,
    metric: str,
    target_block: int = Query(..., description="Target block number"),
    age_weight: float = Query(0.1, ge=0.0, le=1.0, description="Age weight factor"),
    normalize: bool = Query(True, description="Normalize to [0,1]"),
    engine: TemporalCompositeEngine = Depends(get_engine)
):
    """
    Compute age-weighted metric.
    
    Rewards nodes that have been in the network longer.
    """
    from ..models.temporal_composite import TemporalOperationConfig
    
    config = TemporalCompositeConfig(
        name=f"{metric}_age_weighted",
        base_metric=metric,
        temporal_config=TemporalOperationConfig(
            operation=TemporalOperation.AGE_WEIGHTED,
            window_blocks=1,
            age_weight=age_weight,
            normalize_output=normalize
        ),
        base_sql_file=base_sql_file,
        target_block=target_block,
        save=False
    )
    
    try:
        result = await asyncio.to_thread(
            engine.compute_temporal_composite,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))