#!/usr/bin/env python3
"""
Graph Analyzer Web Viewer - Entry Point

Usage:
    python run.py
    
Or with uvicorn directly:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )