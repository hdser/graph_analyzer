#!/usr/bin/env python3
"""
Graph Analyzer Web Viewer - Entry Point

USAGE:

Local Development:
    # From project root directory
    python run.py
    
    # Or with uvicorn directly
    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

Production Deployment (Digital Ocean):
    # If deployed in a directory structure like: /workspace/web_viewer/
    # Run from /workspace/ (parent directory)
    uvicorn web_viewer.backend.main:app --host 0.0.0.0 --port 8080
    
    # Or run from /workspace/web_viewer/ directory
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080

Environment Variables:
    Set these in .env file or environment:
    - DB_USER
    - DB_PASSWORD
    - DB_HOST
    - DB_PORT
    - DB_NAME
"""

import sys
import os
from pathlib import Path

# Add parent directory to path if needed
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import uvicorn

if __name__ == "__main__":
    # Detect if we're in production environment
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    # Use appropriate host based on environment
    host = "0.0.0.0" if is_production else "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting server on {host}:{port}")
    print(f"Environment: {'Production' if is_production else 'Development'}")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=not is_production  # Disable reload in production
    )