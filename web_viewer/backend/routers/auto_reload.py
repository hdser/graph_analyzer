"""
Auto-Reload Router

API endpoints for automatic background reload functionality.
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request

from backend.models.requests import AutoReloadConfig
from backend.services.network_service import network_service
from backend.config import HAS_SSE

if HAS_SSE:
    from sse_starlette.sse import EventSourceResponse


router = APIRouter(prefix="/api/auto-reload", tags=["auto-reload"])


@router.post("/start")
async def start_auto_reload(config: AutoReloadConfig):
    """Start automatic background reloading."""
    if not HAS_SSE:
        raise HTTPException(
            status_code=503, 
            detail="SSE not available. Install sse-starlette."
        )
    
    try:
        status = await network_service.auto_reload_manager.start(config)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_auto_reload():
    """Stop automatic background reloading."""
    try:
        status = await network_service.auto_reload_manager.stop()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_auto_reload_status():
    """Get current auto-reload status."""
    return network_service.auto_reload_manager.get_status()


@router.get("/events")
async def auto_reload_events(request: Request):
    """
    SSE endpoint for real-time reload notifications.
    
    Event types:
    - reload_started
    - reload_progress
    - reload_complete
    - reload_error
    - status_update
    """
    if not HAS_SSE:
        raise HTTPException(
            status_code=503, 
            detail="SSE not available"
        )
    
    async def event_generator():
        queue = network_service.auto_reload_manager.subscribe()
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {
                        "event": event["type"],
                        "data": json.dumps(event["data"])
                    }
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": "{}"}
        finally:
            network_service.auto_reload_manager.unsubscribe(queue)
    
    return EventSourceResponse(event_generator())