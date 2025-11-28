"""
Graph Analyzer Web Viewer - Main Application

FastAPI application with modular router architecture.

Features:
- Network loading from SQL with caching
- Graph visualization with Cytoscape.js
- Multiple layout algorithms (Cytoscape Desktop, layout service, local spring)
- Anomaly detection with 6 algorithms
- Composite metrics creation
- Auto-reload with SSE notifications
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings, print_startup_banner
from backend.routers import (
    network_router,
    metrics_router,
    anomaly_router,
    composite_router,
    auto_reload_router,
)


# Print startup banner
print_startup_banner()


# Create FastAPI application
app = FastAPI(
    title="Graph Analyzer Web Viewer",
    description="Web-based graph visualization and analysis dashboard",
    version="2.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
STATIC_DIR = settings.STATIC_DIR
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Include routers
app.include_router(network_router)
app.include_router(metrics_router)
app.include_router(anomaly_router)
app.include_router(composite_router)
app.include_router(auto_reload_router)


# Root endpoint - serve index.html
@app.get("/")
async def root():
    """Serve the main application page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Graph Analyzer API", "docs": "/docs"}


# Distributions popup
@app.get("/distributions")
async def distributions():
    """Serve the distributions analysis page."""
    dist_path = STATIC_DIR / "distributions.html"
    if dist_path.exists():
        return FileResponse(dist_path)
    return {"error": "distributions.html not found"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)