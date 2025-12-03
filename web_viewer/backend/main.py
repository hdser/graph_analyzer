"""
Graph Analyzer Web Viewer - Main Application

FastAPI application with modular router architecture.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
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
)


print_startup_banner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup: auto-load if HIDE_DATA_SOURCE_UI is enabled
    if settings.HIDE_DATA_SOURCE_UI and settings.DEFAULT_SQL_FILES:
        print("[STARTUP] Production mode - auto-loading...")
        try:
            from .models.requests import LoadConfig
            from .services.network_service import network_service
            
            config = LoadConfig(
                sql_files=settings.DEFAULT_SQL_FILES,
                node_properties_files=settings.DEFAULT_PROPERTIES_FILES,
                use_cached_layout=True,
                skip_sql=False,
                metrics_mode=settings.DEFAULT_METRICS_MODE
            )
            result = network_service.load_network(config)
            print(f"[STARTUP] Loaded {result.node_count} nodes, {result.edge_count} edges")
        except Exception as e:
            print(f"[STARTUP] Auto-load failed: {e}")
            import traceback
            traceback.print_exc()
    
    yield
    
    print("[SHUTDOWN] Graph Analyzer shutting down...")


app = FastAPI(
    title="Graph Analyzer Web Viewer",
    description="Web-based graph visualization and analysis dashboard",
    version="2.0.0",
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


app.include_router(network_router)
app.include_router(metrics_router)
app.include_router(anomaly_router)
app.include_router(composite_router)
app.include_router(auto_reload_router)


@app.get("/")
async def root():
    """Serve the main application page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Graph Analyzer API", "docs": "/docs"}


@app.get("/distributions")
async def distributions():
    """Serve the distributions analysis page."""
    dist_path = STATIC_DIR / "distributions.html"
    if dist_path.exists():
        return FileResponse(dist_path)
    return {"error": "distributions.html not found"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from .services.network_service import network_service
    
    graphs_loaded = len(network_service.graphs) > 0
    node_count = sum(G.number_of_nodes() for G in network_service.graphs.values()) if graphs_loaded else 0
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "mode": "production" if settings.HIDE_DATA_SOURCE_UI else "admin",
        "graphs_loaded": graphs_loaded,
        "node_count": node_count
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)