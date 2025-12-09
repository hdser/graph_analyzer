"""
Configuration Module

Centralized configuration for the Graph Analyzer Web Viewer.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # Base paths
    BASE_DIR: Path = Path(__file__).parent.parent
    SQL_DIR: Path = BASE_DIR.parent / "sql"
    NODE_PROPERTIES_DIR: Path = SQL_DIR / "properties"
    CACHE_DIR: Path = BASE_DIR / "cache"
    LAYOUTS_DIR: Path = CACHE_DIR / "layouts"
    DATA_CACHE_DIR: Path = CACHE_DIR / "data"
    STATIC_DIR: Path = BASE_DIR / "static"
    
    # Database configuration
    DB_USER: str = os.getenv("DB_USER", "readonly_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "circles")
    
    # Layout service
    LAYOUT_SERVICE_URL: str = os.getenv("LAYOUT_SERVICE_URL", "http://localhost:3000/layout")
    
    # Metrics settings
    DEFAULT_METRICS_MODE: str = os.getenv("DEFAULT_METRICS_MODE", "essential")
    N_JOBS: int = int(os.getenv("N_JOBS", "-1"))
    
    # Layout algorithm parameters
    SPRING_STRENGTH: float = float(os.getenv("SPRING_STRENGTH", "0.0008"))
    SPRING_LENGTH: float = float(os.getenv("SPRING_LENGTH", "200"))
    REPULSION_STRENGTH: float = float(os.getenv("REPULSION_STRENGTH", "5000"))
    DAMPING: float = float(os.getenv("DAMPING", "0.9"))
    MAX_VELOCITY: float = float(os.getenv("MAX_VELOCITY", "50"))
    CONVERGENCE_THRESHOLD: float = float(os.getenv("CONVERGENCE_THRESHOLD", "0.5"))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "100"))
    
    # Performance limits
    MAX_NODES_FOR_LOF: int = int(os.getenv("MAX_NODES_FOR_LOF", "50000"))
    MAX_EDGES_FOR_CYTOSCAPE_DESKTOP: int = int(os.getenv("MAX_EDGES_FOR_CYTOSCAPE_DESKTOP", "5000000"))
    EDGE_CHUNK_SIZE: int = int(os.getenv("EDGE_CHUNK_SIZE", "50000"))
    
    # Auto-reload interval (seconds)
    AUTO_RELOAD_MIN_INTERVAL: int = 60
    AUTO_RELOAD_MAX_INTERVAL: int = 3600
    AUTO_RELOAD_DEFAULT_INTERVAL: int = int(os.getenv("AUTO_RELOAD_DEFAULT_INTERVAL", "300"))
    
    # ==========================================================================
    # UI MODE
    # ==========================================================================
    # HIDE_DATA_SOURCE_UI=true  -> Production: auto-load + auto-reload from env
    # HIDE_DATA_SOURCE_UI=false -> Admin: manual control, all menus visible
    # ==========================================================================
    
    HIDE_DATA_SOURCE_UI: bool = os.getenv("HIDE_DATA_SOURCE_UI", "false").lower() == "true"
    
    # Files to load (used when HIDE_DATA_SOURCE_UI=true)
    DEFAULT_SQL_FILES: List[str] = [
        f.strip() for f in os.getenv("DEFAULT_SQL_FILES", "").split(",") if f.strip()
    ]
    DEFAULT_PROPERTIES_FILES: List[str] = [
        f.strip() for f in os.getenv("DEFAULT_PROPERTIES_FILES", "").split(",") if f.strip()
    ]
    
    # ==========================================================================
    # EXTERNAL API PROPERTIES
    # ==========================================================================
    # Configuration for fetching node properties from external REST APIs.
    # Each provider can be enabled/disabled independently.
    # ==========================================================================
    
    # Base URL for external APIs
    EXTERNAL_API_BASE_URL: str = os.getenv(
        "EXTERNAL_API_BASE_URL",
        "https://squid-app-3gxnl.ondigitalocean.app"
    )
    
    # HTTP timeout for API requests (seconds)
    EXTERNAL_API_TIMEOUT: int = int(os.getenv("EXTERNAL_API_TIMEOUT", "30"))
    
    # Number of retries for failed API requests
    EXTERNAL_API_RETRIES: int = int(os.getenv("EXTERNAL_API_RETRIES", "3"))
    
    # Cache TTL for API properties (seconds, 0 = no cache expiry)
    EXTERNAL_API_CACHE_TTL: int = int(os.getenv("EXTERNAL_API_CACHE_TTL", "3600"))
    
    # Comma-separated list of enabled providers
    EXTERNAL_API_PROVIDERS: List[str] = [
        p.strip() for p in os.getenv("EXTERNAL_API_PROVIDERS", "blacklist").split(",") if p.strip()
    ]
    
    # --------------------------------------------------------------------------
    # Blacklist Provider Settings
    # --------------------------------------------------------------------------
    EXTERNAL_API_BLACKLIST_ENABLED: bool = os.getenv(
        "EXTERNAL_API_BLACKLIST_ENABLED", "true"
    ).lower() == "true"
    
    EXTERNAL_API_BLACKLIST_ENDPOINT: str = os.getenv(
        "EXTERNAL_API_BLACKLIST_ENDPOINT",
        "/aboutcircles-advanced-analytics2/bot-analytics/blacklist"
    )
    
    # Whether to filter for v2 addresses only
    EXTERNAL_API_BLACKLIST_V2_ONLY: bool = os.getenv(
        "EXTERNAL_API_BLACKLIST_V2_ONLY", "true"
    ).lower() == "true"
    
    # --------------------------------------------------------------------------
    # Add new provider settings here following the pattern:
    # EXTERNAL_API_{PROVIDER}_ENABLED
    # EXTERNAL_API_{PROVIDER}_ENDPOINT
    # EXTERNAL_API_{PROVIDER}_{SETTING}
    # --------------------------------------------------------------------------
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.NODE_PROPERTIES_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()


# Feature availability flags
try:
    from sse_starlette.sse import EventSourceResponse
    HAS_SSE = True
except ImportError:
    HAS_SSE = False

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import py4cytoscape as p4c
    HAS_CYTOSCAPE_DESKTOP = True
except ImportError:
    HAS_CYTOSCAPE_DESKTOP = False

HAS_ANOMALY = HAS_SKLEARN


def print_startup_banner():
    """Print startup information banner."""
    print("\n" + "=" * 60)
    print("  Graph Analyzer Web Viewer v2.0.0")
    print("=" * 60)
    print(f"  Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    print(f"  SQL Dir:  {settings.SQL_DIR}")
    print(f"  Properties Dir: {settings.NODE_PROPERTIES_DIR}")
    print(f"  Cache:    {settings.CACHE_DIR}")
    print("-" * 60)
    print(f"  SSE Support:        {'Y' if HAS_SSE else 'N'}")
    print(f"  Anomaly Detection:  {'Y' if HAS_ANOMALY else 'N'}")
    print(f"  Cytoscape Desktop:  {'Y' if HAS_CYTOSCAPE_DESKTOP else 'N'}")
    print("-" * 60)
    # External API providers
    print("  External API Properties:")
    print(f"    Base URL: {settings.EXTERNAL_API_BASE_URL}")
    print(f"    Providers: {', '.join(settings.EXTERNAL_API_PROVIDERS) or 'None'}")
    print(f"    Blacklist: {'Y' if settings.EXTERNAL_API_BLACKLIST_ENABLED else 'N'}")
    print("-" * 60)
    if settings.HIDE_DATA_SOURCE_UI:
        print("  Mode: Production (auto-load + auto-reload)")
        print(f"    SQL Files:      {', '.join(settings.DEFAULT_SQL_FILES) or 'None'}")
        print(f"    Properties:     {', '.join(settings.DEFAULT_PROPERTIES_FILES) or 'None'}")
        print(f"    Reload Interval: {settings.AUTO_RELOAD_DEFAULT_INTERVAL}s")
    else:
        print("  Mode: Admin (manual control)")
    print("=" * 60 + "\n")