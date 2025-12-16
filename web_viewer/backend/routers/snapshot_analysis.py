"""
Snapshot Analysis Router

API endpoints for analyzing historical snapshots including:
- Running analysis on snapshots
- Batch analysis with progress
- Retrieving analysis results
- Getting specific metric values
"""

import json
import asyncio
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from ..config import HAS_SSE
from ..models.snapshot_analysis import (
    SnapshotAnalysisConfig,
    SnapshotAnalysisResult,
    BatchAnalysisConfig,
    BatchAnalysisResult,
    AnalysisProgressUpdate,
    AnalysisStatus,
    AnalyzedSnapshotsListResponse,
    MetricValuesResponse,
)
from ..services.snapshot_analysis_service import (
    SnapshotAnalysisService,
    snapshot_analysis_service
)


router = APIRouter(prefix="/api/snapshots", tags=["snapshot-analysis"])


def get_analysis_service() -> SnapshotAnalysisService:
    """Get snapshot analysis service instance."""
    return snapshot_analysis_service


def parse_snapshot_id(snapshot_id: str) -> tuple:
    """
    Parse snapshot_id into (base_sql_file, block_number).
    
    Format: {base_sql_file}_block_{block_number}
    """
    parts = snapshot_id.rsplit("_block_", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid snapshot_id format: {snapshot_id}"
        )
    
    base_sql_file = parts[0]
    try:
        block_number = int(parts[1])
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid block number in snapshot_id: {snapshot_id}"
        )
    
    return base_sql_file, block_number


# =============================================================================
# Analysis Endpoints
# =============================================================================

@router.post("/{snapshot_id}/analyze", response_model=SnapshotAnalysisResult)
async def analyze_snapshot(
    snapshot_id: str,
    config: SnapshotAnalysisConfig = None,
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Run analysis on a single snapshot.
    
    Computes metrics and optionally runs anomaly detection on the snapshot data.
    Results are stored for future retrieval.
    
    Args:
        snapshot_id: Snapshot identifier (format: {base_sql_file}_block_{block_number})
        config: Analysis configuration (uses defaults if not provided)
    
    Returns:
        SnapshotAnalysisResult with all analysis data
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    if config is None:
        config = SnapshotAnalysisConfig()
    
    try:
        # Run analysis in thread pool to not block
        result = await asyncio.to_thread(
            service.analyze_snapshot,
            base_sql_file,
            block_number,
            config
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze-batch")
async def analyze_batch(
    config: BatchAnalysisConfig,
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Run analysis on multiple snapshots with SSE progress reporting.
    
    Returns Server-Sent Events stream with progress updates.
    Event types:
    - progress: Current snapshot being analyzed
    - complete: Individual snapshot analysis complete
    - error: Error for individual snapshot
    - done: All snapshots analyzed
    
    If SSE is not available, returns JSON result after all analyses complete.
    """
    if not HAS_SSE:
        # Fallback: run all analyses and return JSON
        results = []
        failed = []
        
        for block_number in config.block_numbers:
            try:
                result = await asyncio.to_thread(
                    service.analyze_snapshot,
                    config.base_sql_file,
                    block_number,
                    config.config
                )
                results.append(result)
            except Exception as e:
                failed.append({"block_number": block_number, "error": str(e)})
        
        return BatchAnalysisResult(
            base_sql_file=config.base_sql_file,
            total_requested=len(config.block_numbers),
            total_completed=len(results),
            total_failed=len(failed),
            results=results,
            failed_snapshots=failed,
            total_computation_time_seconds=sum(r.computation_time_seconds for r in results)
        )
    
    from sse_starlette.sse import EventSourceResponse
    
    async def generate():
        total = len(config.block_numbers)
        results = []
        failed = []
        start_time = asyncio.get_event_loop().time()
        
        # Sort block numbers ascending
        block_numbers = sorted(config.block_numbers)
        
        for idx, block_number in enumerate(block_numbers, 1):
            snapshot_id = f"{config.base_sql_file}_block_{block_number}"
            
            try:
                # Progress event
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "snapshot_id": snapshot_id,
                        "block_number": block_number,
                        "status": "analyzing"
                    })
                }
                
                # Run analysis
                result = await asyncio.to_thread(
                    service.analyze_snapshot,
                    config.base_sql_file,
                    block_number,
                    config.config
                )
                
                results.append(result)
                
                # Complete event
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "current": idx,
                        "total": total,
                        "snapshot_id": snapshot_id,
                        "status": result.status.value,
                        "metrics_count": len(result.metrics_computed),
                        "anomaly_count": result.anomaly_results.anomaly_count if result.anomaly_results else None,
                        "computation_time": result.computation_time_seconds
                    })
                }
                
            except Exception as e:
                failed.append({"block_number": block_number, "error": str(e)})
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "snapshot_id": snapshot_id,
                        "block_number": block_number,
                        "error": str(e)
                    })
                }
        
        # Done event
        total_time = asyncio.get_event_loop().time() - start_time
        yield {
            "event": "done",
            "data": json.dumps({
                "total_completed": len(results),
                "total_failed": len(failed),
                "total_computation_time": total_time,
                "snapshot_ids": [r.snapshot_id for r in results]
            })
        }
    
    return EventSourceResponse(generate())


# =============================================================================
# Results Retrieval Endpoints
# =============================================================================

@router.get("/{snapshot_id}/analysis", response_model=SnapshotAnalysisResult)
async def get_analysis_results(
    snapshot_id: str,
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Get stored analysis results for a snapshot.
    
    Returns the analysis results if they exist, 404 if no analysis has been run.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    result = await asyncio.to_thread(
        service.load_analysis_results,
        base_sql_file,
        block_number
    )
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for snapshot: {snapshot_id}"
        )
    
    return result


@router.get("/{snapshot_id}/metrics/{metric_name}")
async def get_snapshot_metric_values(
    snapshot_id: str,
    metric_name: str,
    include_values: bool = Query(True, description="Include per-node values (can be large)"),
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Get values for a specific metric from a snapshot's analysis.
    
    Returns statistics and optionally per-node values for the requested metric.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    result = await asyncio.to_thread(
        service.get_metric_values,
        base_sql_file,
        block_number,
        metric_name,
        include_values
    )
    
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metric '{metric_name}' not found in snapshot {snapshot_id}"
        )
    
    return result


@router.get("/analyzed", response_model=AnalyzedSnapshotsListResponse)
async def list_analyzed_snapshots(
    base_sql_file: str = Query(..., description="Base SQL file name"),
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    List all snapshots that have analysis results.
    
    Returns information about each analyzed snapshot including
    which metrics were computed and anomaly detection results.
    """
    analyzed = await asyncio.to_thread(
        service.list_analyzed_snapshots,
        base_sql_file
    )
    
    return AnalyzedSnapshotsListResponse(
        base_sql_file=base_sql_file,
        snapshots=analyzed,
        total_count=len(analyzed)
    )


@router.get("/{snapshot_id}/has-analysis")
async def check_has_analysis(
    snapshot_id: str,
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Quick check if a snapshot has analysis results.
    
    Returns a simple boolean response.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    has_analysis = await asyncio.to_thread(
        service.has_analysis,
        base_sql_file,
        block_number
    )
    
    return {"snapshot_id": snapshot_id, "has_analysis": has_analysis}


# =============================================================================
# Management Endpoints
# =============================================================================

@router.delete("/{snapshot_id}/analysis")
async def delete_analysis(
    snapshot_id: str,
    service: SnapshotAnalysisService = Depends(get_analysis_service)
):
    """
    Delete stored analysis results for a snapshot.
    
    This removes the analysis data but keeps the snapshot itself.
    """
    base_sql_file, block_number = parse_snapshot_id(snapshot_id)
    
    success = await asyncio.to_thread(
        service.delete_analysis,
        base_sql_file,
        block_number
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found to delete for snapshot: {snapshot_id}"
        )
    
    return {"success": True, "snapshot_id": snapshot_id}


# =============================================================================
# Utility Endpoints
# =============================================================================

@router.get("/analysis/algorithms")
async def get_available_algorithms():
    """
    Get available anomaly detection algorithms for analysis.
    """
    from ..config import HAS_ANOMALY
    
    if not HAS_ANOMALY:
        return {"algorithms": [], "available": False}
    
    try:
        from engines.anomaly_engine import AnomalyEngine
        engine = AnomalyEngine()
        algorithms = engine.get_available_algorithms()
        
        return {
            "available": True,
            "algorithms": {
                name: {
                    "name": info.name,
                    "display_name": info.display_name,
                    "description": info.description,
                    "multivariate": info.supports_multivariate,
                    "default_parameters": info.default_parameters
                }
                for name, info in algorithms.items()
            }
        }
    except Exception as e:
        return {"algorithms": [], "available": False, "error": str(e)}


@router.get("/analysis/metrics-modes")
async def get_metrics_modes():
    """
    Get available metrics computation modes.
    """
    try:
        from engines.graph_metrics import METRIC_PRESETS, METRIC_CATEGORIES
        
        return {
            "presets": METRIC_PRESETS,
            "categories": METRIC_CATEGORIES,
            "default": "essential"
        }
    except Exception as e:
        return {
            "presets": {
                "basic": ["topology", "community"],
                "essential": ["topology", "centrality", "clustering", "community"]
            },
            "categories": {},
            "error": str(e)
        }