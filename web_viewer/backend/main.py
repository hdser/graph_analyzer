"""
Graph Analyzer Web Viewer - Main Application

FastAPI application with modular router architecture.

Location: web_viewer/backend/main.py
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import settings, print_startup_banner
from .routers import (
    network_router,
    metrics_router,
    anomaly_router,
    composite_router,
    auto_reload_router,
    snapshots_router,
    snapshot_analysis_router,
    timeseries_router,
    temporal_composite_router,
    graph_algorithms_router,
    capacity_flow_router,
    embeddings_router,
    query_router,
)


print_startup_banner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    import asyncio
    
    # Startup: schedule auto-load if enabled and SQL files configured
    if settings.AUTO_LOAD_ON_STARTUP and settings.DEFAULT_SQL_FILES:
        print("[STARTUP] Auto-load enabled - scheduling background data load...")
        
        async def background_auto_load():
            """Load network data in background after server starts."""
            # Small delay to ensure server is fully ready
            await asyncio.sleep(2)
            
            try:
                from .models.requests import LoadConfig
                from .services.network_service import network_service
                
                print("[STARTUP] Starting background data load...")
                startup_status["status"] = "loading"
                startup_status["message"] = "Loading SQL files..."
                
                # Notify waiting clients
                await notify_startup_subscribers()
                
                config = LoadConfig(
                    sql_files=settings.DEFAULT_SQL_FILES,
                    node_properties_files=settings.DEFAULT_PROPERTIES_FILES,
                    use_cached_layout=True,
                    skip_sql=False,
                    metrics_mode=settings.DEFAULT_METRICS_MODE
                )
                
                # Run the blocking load in a thread to not block the event loop
                result = await asyncio.to_thread(network_service.load_network, config)
                
                # Update status to ready
                startup_status["status"] = "ready"
                startup_status["message"] = f"Loaded {result.node_count} nodes, {result.edge_count} edges"
                startup_status["node_count"] = result.node_count
                startup_status["edge_count"] = result.edge_count
                startup_status["loaded_graphs"] = result.loaded_graphs
                
                print(f"[STARTUP] Background load complete: {result.node_count} nodes, {result.edge_count} edges")
                
                # Notify waiting clients
                await notify_startup_subscribers()
                
            except Exception as e:
                startup_status["status"] = "error"
                startup_status["message"] = str(e)
                print(f"[STARTUP] Background auto-load failed: {e}")
                import traceback
                traceback.print_exc()
                await notify_startup_subscribers()
        
        # Schedule the background task
        asyncio.create_task(background_auto_load())
    
    yield
    
    print("[SHUTDOWN] Graph Analyzer shutting down...")
    try:
        from .services.network_service import network_service
        network_service.flush_all_cosmos_positions()
    except Exception as e:
        print(f"[SHUTDOWN] Failed to flush cosmos positions: {e}")


# Startup status tracking for SSE
startup_status = {
    "status": "idle",  # idle, loading, ready, error
    "message": "",
    "node_count": 0,
    "edge_count": 0,
    "loaded_graphs": []
}
startup_subscribers = []


async def notify_startup_subscribers():
    """Notify all SSE subscribers of startup status change."""
    for queue in startup_subscribers:
        try:
            await queue.put(startup_status.copy())
        except:
            pass


app = FastAPI(
    title="Graph Analyzer Web Viewer",
    description="Web-based graph visualization and analysis dashboard",
    version="2.2.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = settings.STATIC_DIR
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

NO_CACHE_HTML_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


# Register routers
app.include_router(network_router)
app.include_router(metrics_router)
app.include_router(anomaly_router)
app.include_router(composite_router)
app.include_router(auto_reload_router)
app.include_router(snapshots_router)
app.include_router(snapshot_analysis_router)
app.include_router(timeseries_router)
app.include_router(temporal_composite_router)
app.include_router(graph_algorithms_router)
app.include_router(capacity_flow_router)
app.include_router(embeddings_router)
app.include_router(query_router)


@app.get("/")
async def root():
    """Serve the main application page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, headers=NO_CACHE_HTML_HEADERS)
    return {"message": "Graph Analyzer API", "docs": "/docs"}


@app.get("/distributions")
async def distributions():
    """Serve the distributions analysis page."""
    dist_path = STATIC_DIR / "distributions.html"
    if dist_path.exists():
        return FileResponse(dist_path, headers=NO_CACHE_HTML_HEADERS)
    return {"error": "distributions.html not found"}


@app.get("/data-explorer")
async def data_explorer():
    """Serve the data explorer page."""
    explorer_path = STATIC_DIR / "data-explorer.html"
    if explorer_path.exists():
        return FileResponse(explorer_path, headers=NO_CACHE_HTML_HEADERS)
    return {"error": "data-explorer.html not found"}


@app.get("/sql-explorer")
async def sql_explorer():
    """Serve the SQL explorer page."""
    sql_path = STATIC_DIR / "sql-explorer.html"
    if sql_path.exists():
        return FileResponse(sql_path, headers=NO_CACHE_HTML_HEADERS)
    return {"error": "sql-explorer.html not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from .services.network_service import network_service
    
    # Check deep learning availability
    deep_learning_available = False
    try:
        from engines.deep_learning import HAS_DEEP_LEARNING
        deep_learning_available = HAS_DEEP_LEARNING
    except ImportError:
        pass
    
    graphs_loaded = len(network_service.graphs) > 0
    node_count = sum(G.number_of_nodes() for G in network_service.graphs.values()) if graphs_loaded else 0
    
    # Check if background loading is expected but not complete
    loading_expected = settings.AUTO_LOAD_ON_STARTUP and settings.DEFAULT_SQL_FILES
    loading_status = "ready" if graphs_loaded else ("loading" if loading_expected else "idle")
    
    return {
        "status": "healthy",
        "version": "2.2.0",
        "mode": "production" if settings.PRODUCTION_MODE else "admin",
        "data_status": loading_status,
        "graphs_loaded": graphs_loaded,
        "node_count": node_count,
        "deep_learning_available": deep_learning_available
    }


@app.get("/api/startup-status")
async def get_startup_status():
    """Get current startup loading status (for initial check)."""
    return startup_status


@app.get("/api/startup-events")
async def startup_events(request: Request):
    """
    SSE endpoint for startup loading status.
    Connect once and receive events when loading completes.
    """
    import asyncio
    from .config import HAS_SSE
    
    if not HAS_SSE:
        # Fallback: just return current status
        return startup_status
    
    from sse_starlette.sse import EventSourceResponse
    
    async def event_generator():
        queue = asyncio.Queue()
        startup_subscribers.append(queue)
        
        try:
            # Send current status immediately
            yield {
                "event": "status",
                "data": json.dumps(startup_status)
            }
            
            # If already ready, close connection
            if startup_status["status"] == "ready":
                return
            
            # Wait for status updates
            while True:
                try:
                    # Wait for update with timeout
                    status = await asyncio.wait_for(queue.get(), timeout=30)
                    yield {
                        "event": "status", 
                        "data": json.dumps(status)
                    }
                    
                    # If ready or error, close connection
                    if status["status"] in ("ready", "error"):
                        return
                        
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"event": "ping", "data": ""}
                    
                # Check if client disconnected
                if await request.is_disconnected():
                    break
                    
        finally:
            if queue in startup_subscribers:
                startup_subscribers.remove(queue)
    
    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
