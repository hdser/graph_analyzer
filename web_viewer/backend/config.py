"""
Configuration Module

Centralized configuration for the Graph Analyzer Web Viewer.
All settings are loaded from environment variables with sensible defaults.

Location: web_viewer/backend/config.py
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
    SNAPSHOT_SQL_DIR: Path = SQL_DIR / "snapshots"
    CACHE_DIR: Path = BASE_DIR / "cache"
    LAYOUTS_DIR: Path = CACHE_DIR / "layouts"
    DATA_CACHE_DIR: Path = CACHE_DIR / "data"
    STATIC_DIR: Path = BASE_DIR / "static"
    
    # ==========================================================================
    # RENDERER CONFIGURATION
    # ==========================================================================
    
    # Default renderer preference: 'auto', 'cosmos', 'cytoscape'
    # auto: Select based on graph size and WebGL capabilities
    # cosmos: Force cosmos.gl (falls back to cytoscape if unavailable)
    # cytoscape: Force Cytoscape.js
    RENDERER_PREFERENCE: str = os.getenv("RENDERER_PREFERENCE", "auto")
    
    # Node count thresholds for automatic renderer selection
    # Below this threshold, Cytoscape.js is used (more feature-rich for small graphs)
    RENDERER_COSMOS_MIN_NODES: int = int(os.getenv("RENDERER_COSMOS_MIN_NODES", "5000"))
    # Above this threshold, cosmos.gl is strongly preferred (Cytoscape may be slow)
    RENDERER_COSMOS_PREFERRED_NODES: int = int(os.getenv("RENDERER_COSMOS_PREFERRED_NODES", "50000"))
    
    # cosmos.gl specific settings
    COSMOS_SPACE_SIZE: int = int(os.getenv("COSMOS_SPACE_SIZE", "8192"))
    COSMOS_POINT_SIZE: float = float(os.getenv("COSMOS_POINT_SIZE", "6"))
    COSMOS_LINK_WIDTH: float = float(os.getenv("COSMOS_LINK_WIDTH", "1"))
    COSMOS_BACKGROUND_COLOR: str = os.getenv("COSMOS_BACKGROUND_COLOR", "#1a1a1a")
    COSMOS_CURVED_LINKS: bool = os.getenv("COSMOS_CURVED_LINKS", "true").lower() == "true"
    COSMOS_ENABLE_DRAG: bool = os.getenv("COSMOS_ENABLE_DRAG", "true").lower() == "true"
    
    # cosmos.gl simulation parameters
    COSMOS_SIMULATION_FRICTION: float = float(os.getenv("COSMOS_SIMULATION_FRICTION", "0.85"))
    COSMOS_SIMULATION_GRAVITY: float = float(os.getenv("COSMOS_SIMULATION_GRAVITY", "0.1"))
    COSMOS_SIMULATION_REPULSION: float = float(os.getenv("COSMOS_SIMULATION_REPULSION", "0.5"))
    COSMOS_SIMULATION_LINK_DISTANCE: float = float(os.getenv("COSMOS_SIMULATION_LINK_DISTANCE", "10"))
    COSMOS_SIMULATION_LINK_SPRING: float = float(os.getenv("COSMOS_SIMULATION_LINK_SPRING", "0.3"))
    
    # Node size range for metric-based sizing
    RENDERER_NODE_SIZE_MIN: float = float(os.getenv("RENDERER_NODE_SIZE_MIN", "5"))
    RENDERER_NODE_SIZE_MAX: float = float(os.getenv("RENDERER_NODE_SIZE_MAX", "30"))
    
    # Default color gradient
    RENDERER_DEFAULT_GRADIENT: str = os.getenv("RENDERER_DEFAULT_GRADIENT", "spectral")
    
    # Selection colors (RGBA, 0-1 range for cosmos.gl, hex for cytoscape)
    RENDERER_SELECTION_COLOR: str = os.getenv("RENDERER_SELECTION_COLOR", "#FF0000")
    RENDERER_HIGHLIGHT_COLOR: str = os.getenv("RENDERER_HIGHLIGHT_COLOR", "#FFA500")
    RENDERER_DEFAULT_NODE_COLOR: str = os.getenv("RENDERER_DEFAULT_NODE_COLOR", "#999999")
    RENDERER_DEFAULT_EDGE_COLOR: str = os.getenv("RENDERER_DEFAULT_EDGE_COLOR", "#f0f0f0")
    RENDERER_DEFAULT_EDGE_OPACITY: float = float(os.getenv("RENDERER_DEFAULT_EDGE_OPACITY", "0.3"))
    
    # ==========================================================================
    # SNAPSHOT CONFIGURATION
    # ==========================================================================
    
    SNAPSHOT_CACHE_DIR: Path = CACHE_DIR / "snapshots"
    SNAPSHOT_MASTER_LAYOUTS_DIR: str = "_master_layouts"
    SNAPSHOT_INDEX_FILE: str = "_index.json"
    
    SNAPSHOT_SPRING_ITERATIONS: int = int(os.getenv("SNAPSHOT_SPRING_ITERATIONS", "50"))
    SNAPSHOT_SPRING_K: float = float(os.getenv("SNAPSHOT_SPRING_K", "1.0"))
    SNAPSHOT_SPRING_REPULSION: float = float(os.getenv("SNAPSHOT_SPRING_REPULSION", "100.0"))
    SNAPSHOT_SPRING_ATTRACTION: float = float(os.getenv("SNAPSHOT_SPRING_ATTRACTION", "0.1"))
    SNAPSHOT_SPRING_DAMPING: float = float(os.getenv("SNAPSHOT_SPRING_DAMPING", "0.9"))
    
    SNAPSHOT_DEFAULT_METRICS_MODE: str = os.getenv("SNAPSHOT_DEFAULT_METRICS_MODE", "standard")
    
    SNAPSHOT_STANDARD_METRICS: List[str] = [
        "in_degree", "out_degree", "total_degree",
        "pagerank", "betweenness_centrality"
    ]
    
    SNAPSHOT_MAX_BATCH_SIZE: int = int(os.getenv("SNAPSHOT_MAX_BATCH_SIZE", "30"))
    SNAPSHOT_MAX_SUGGESTIONS: int = int(os.getenv("SNAPSHOT_MAX_SUGGESTIONS", "90"))
    
    # ==========================================================================
    # DATABASE CONFIGURATION
    # ==========================================================================
    
    DB_USER: str = os.getenv("DB_USER", "readonly_user")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "circles")
    
    # Metrics settings
    DEFAULT_METRICS_MODE: str = os.getenv("DEFAULT_METRICS_MODE", "essential")
    N_JOBS: int = int(os.getenv("N_JOBS", "-1"))
    
    # ==========================================================================
    # LAYOUT BACKEND CONFIGURATION
    # ==========================================================================
    
    # Priority order for layout backends (comma-separated)
    # Available: cached, cytoscape_desktop, igraph, fa2, local_spring, circular
    # Note: layout_service (Node.js) has been deprecated and removed
    LAYOUT_BACKEND_PRIORITY: List[str] = [
        p.strip() for p in os.getenv(
            "LAYOUT_BACKEND_PRIORITY",
            "cached,cytoscape_desktop,igraph,fa2,local_spring,circular"
        ).split(",") if p.strip() and p.strip() != "layout_service"
    ]
    
    # igraph settings
    IGRAPH_DEFAULT_ALGORITHM: str = os.getenv("IGRAPH_DEFAULT_ALGORITHM", "auto")
    IGRAPH_SCALE: float = float(os.getenv("IGRAPH_SCALE", "1000.0"))
    
    # ForceAtlas2 settings
    FA2_ITERATIONS: int = int(os.getenv("FA2_ITERATIONS", "1000"))
    FA2_BARNES_HUT_OPTIMIZE: bool = os.getenv("FA2_BARNES_HUT_OPTIMIZE", "true").lower() == "true"
    FA2_BARNES_HUT_THETA: float = float(os.getenv("FA2_BARNES_HUT_THETA", "1.2"))
    FA2_SCALING_RATIO: float = float(os.getenv("FA2_SCALING_RATIO", "2.0"))
    FA2_GRAVITY: float = float(os.getenv("FA2_GRAVITY", "1.0"))
    FA2_SCALE: float = float(os.getenv("FA2_SCALE", "1.0"))
    
    # ==========================================================================
    # LOCAL SPRING LAYOUT PARAMETERS
    # ==========================================================================
    
    SPRING_STRENGTH: float = float(os.getenv("SPRING_STRENGTH", "0.0008"))
    SPRING_LENGTH: float = float(os.getenv("SPRING_LENGTH", "200"))
    REPULSION_STRENGTH: float = float(os.getenv("REPULSION_STRENGTH", "5000"))
    DAMPING: float = float(os.getenv("DAMPING", "0.9"))
    MAX_VELOCITY: float = float(os.getenv("MAX_VELOCITY", "50"))
    CONVERGENCE_THRESHOLD: float = float(os.getenv("CONVERGENCE_THRESHOLD", "0.5"))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "100"))
    
    # ==========================================================================
    # PERFORMANCE LIMITS
    # ==========================================================================
    
    MAX_NODES_FOR_LOF: int = int(os.getenv("MAX_NODES_FOR_LOF", "50000"))
    MAX_EDGES_FOR_CYTOSCAPE_DESKTOP: int = int(os.getenv("MAX_EDGES_FOR_CYTOSCAPE_DESKTOP", "5000000"))
    MAX_NODES_FOR_LOCAL_SPRING: int = int(os.getenv("MAX_NODES_FOR_LOCAL_SPRING", "10000"))
    EDGE_CHUNK_SIZE: int = int(os.getenv("EDGE_CHUNK_SIZE", "50000"))
    
    # Auto-reload interval (seconds)
    AUTO_RELOAD_MIN_INTERVAL: int = 60
    AUTO_RELOAD_MAX_INTERVAL: int = 3600
    AUTO_RELOAD_DEFAULT_INTERVAL: int = int(os.getenv("AUTO_RELOAD_DEFAULT_INTERVAL", "300"))
    
    # ==========================================================================
    # UI MODE CONFIGURATION
    # ==========================================================================
    # 
    # PRODUCTION_MODE: Master toggle for production vs admin mode
    #   - false (default): Admin mode - all panels visible, manual control
    #   - true: Production mode - applies UI_HIDDEN_PANELS, enables auto-load
    #
    # UI_HIDDEN_PANELS: Comma-separated list of panels to hide in production mode
    #   Available panels: load, reload, snapshots, metrics, paths, subgraph,
    #                     flow, embeddings, settings, data-explorer
    #
    PRODUCTION_MODE: bool = os.getenv("PRODUCTION_MODE", "false").lower() == "true"
    
    UI_HIDDEN_PANELS: List[str] = [
        p.strip() for p in os.getenv(
            "UI_HIDDEN_PANELS",
            "load,reload"
        ).split(",") if p.strip()
    ]
    
    AVAILABLE_PANELS: List[str] = [
        "load", "reload", "snapshots", "metrics", "paths", 
        "subgraph", "flow", "filter", "style", "layout", "embeddings"
    ]
    
    # Auto-load configuration (used when PRODUCTION_MODE=true)
    # Defaults to True when in production mode
    AUTO_LOAD_ON_STARTUP: bool = (
        os.getenv("AUTO_LOAD_ON_STARTUP", "true" if PRODUCTION_MODE else "false").lower() == "true"
    )
    
    # Default SQL files to auto-load (comma-separated list)
    DEFAULT_SQL_FILES: List[str] = [
        f.strip() for f in os.getenv(
            "DEFAULT_SQL_FILES",
            ""
        ).split(",") if f.strip()
    ]
    
    # Default node properties files to auto-load (comma-separated list)
    DEFAULT_PROPERTIES_FILES: List[str] = [
        f.strip() for f in os.getenv(
            "DEFAULT_PROPERTIES_FILES",
            ""
        ).split(",") if f.strip()
    ]
    
    # ==========================================================================
    # EXTERNAL API PROPERTIES CONFIGURATION
    # ==========================================================================
    
    # Base URL for external API (v1/v2 endpoints)
    EXTERNAL_API_BASE_URL: str = os.getenv(
        "EXTERNAL_API_BASE_URL",
        "https://squid-app-3gxnl.ondigitalocean.app"
    )
    
    # HTTP request settings
    EXTERNAL_API_TIMEOUT: int = int(os.getenv("EXTERNAL_API_TIMEOUT", "30"))
    EXTERNAL_API_RETRIES: int = int(os.getenv("EXTERNAL_API_RETRIES", "3"))
    EXTERNAL_API_CACHE_TTL: int = int(os.getenv("EXTERNAL_API_CACHE_TTL", "3600"))  # 1 hour default
    
    # Enabled external API providers (comma-separated)
    # Available: blacklist
    EXTERNAL_API_PROVIDERS: List[str] = [
        p.strip() for p in os.getenv(
            "EXTERNAL_API_PROVIDERS",
            ""
        ).split(",") if p.strip()
    ]
    
    # Blacklist-specific settings
    EXTERNAL_API_BLACKLIST_ENABLED: bool = "blacklist" in os.getenv("EXTERNAL_API_PROVIDERS", "")
    
    EXTERNAL_API_BLACKLIST_ENDPOINT: str = os.getenv(
        "EXTERNAL_API_BLACKLIST_ENDPOINT",
        "/aboutcircles-advanced-analytics2/bot-analytics/blacklist"
    )
    
    EXTERNAL_API_BLACKLIST_V2_ONLY: bool = os.getenv(
        "EXTERNAL_API_BLACKLIST_V2_ONLY", "true"
    ).lower() == "true"
    
    # ==========================================================================
    # CAPACITY FLOW CONFIGURATION
    # ==========================================================================
    
    # Router address for Circles protocol
    CIRCLES_ROUTER_ADDRESS: str = os.getenv(
        "CIRCLES_ROUTER_ADDRESS",
        "0xdc287474114cc0551a81ddc2eb51783fbf34802f"
    )
    
    # Priority order for capacity flow backends
    # Available: ortools, networkx
    CAPACITY_FLOW_BACKEND_PRIORITY: List[str] = [
        p.strip() for p in os.getenv(
            "CAPACITY_FLOW_BACKEND_PRIORITY",
            "ortools,networkx"
        ).split(",") if p.strip()
    ]
    
    # Default algorithm for NetworkX backend
    CAPACITY_FLOW_NETWORKX_ALGORITHM: str = os.getenv(
        "CAPACITY_FLOW_NETWORKX_ALGORITHM",
        "preflow_push"
    )
    
    # Cache settings for capacity graphs
    CAPACITY_FLOW_CACHE_ENABLED: bool = os.getenv(
        "CAPACITY_FLOW_CACHE_ENABLED", "true"
    ).lower() == "true"
    
    CAPACITY_FLOW_CACHE_TTL: int = int(os.getenv("CAPACITY_FLOW_CACHE_TTL", "3600"))
    
    @property
    def database_url(self) -> str:
        """Construct database URL from components."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    def get_renderer_config(self) -> dict:
        """Get renderer configuration as dictionary for frontend."""
        return {
            "preference": self.RENDERER_PREFERENCE,
            "thresholds": {
                "cosmosMinNodes": self.RENDERER_COSMOS_MIN_NODES,
                "cosmosPreferredNodes": self.RENDERER_COSMOS_PREFERRED_NODES
            },
            "cosmos": {
                "spaceSize": self.COSMOS_SPACE_SIZE,
                "pointSize": self.COSMOS_POINT_SIZE,
                "linkWidth": self.COSMOS_LINK_WIDTH,
                "backgroundColor": self.COSMOS_BACKGROUND_COLOR,
                "curvedLinks": self.COSMOS_CURVED_LINKS,
                "enableDrag": self.COSMOS_ENABLE_DRAG,
                "simulation": {
                    "friction": self.COSMOS_SIMULATION_FRICTION,
                    "gravity": self.COSMOS_SIMULATION_GRAVITY,
                    "repulsion": self.COSMOS_SIMULATION_REPULSION,
                    "linkDistance": self.COSMOS_SIMULATION_LINK_DISTANCE,
                    "linkSpring": self.COSMOS_SIMULATION_LINK_SPRING
                }
            },
            "style": {
                "nodeSizeMin": self.RENDERER_NODE_SIZE_MIN,
                "nodeSizeMax": self.RENDERER_NODE_SIZE_MAX,
                "defaultGradient": self.RENDERER_DEFAULT_GRADIENT,
                "selectionColor": self.RENDERER_SELECTION_COLOR,
                "highlightColor": self.RENDERER_HIGHLIGHT_COLOR,
                "defaultNodeColor": self.RENDERER_DEFAULT_NODE_COLOR,
                "defaultEdgeColor": self.RENDERER_DEFAULT_EDGE_COLOR,
                "defaultEdgeOpacity": self.RENDERER_DEFAULT_EDGE_OPACITY
            }
        }
    
    def ensure_directories(self):
        """Create required directories if they don't exist."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.NODE_PROPERTIES_DIR.mkdir(parents=True, exist_ok=True)
        self.SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.SNAPSHOT_SQL_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()


# ==========================================================================
# FEATURE AVAILABILITY FLAGS
# ==========================================================================

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

try:
    import igraph
    HAS_IGRAPH = True
except ImportError:
    HAS_IGRAPH = False

try:
    # Try fa2_modified first (maintained fork, works with Python 3.9+)
    from fa2_modified import ForceAtlas2
    HAS_FA2 = True
except ImportError:
    try:
        # Fallback to original fa2
        from fa2 import ForceAtlas2
        HAS_FA2 = True
    except ImportError:
        HAS_FA2 = False

try:
    from ortools.graph.python import max_flow
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

HAS_ANOMALY = HAS_SKLEARN


def print_startup_banner():
    """Print startup information banner."""
    print("\n" + "=" * 60)
    print("  Graph Analyzer Web Viewer v2.4.0")
    print("=" * 60)
    print(f"  Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    print(f"  SQL Dir:  {settings.SQL_DIR}")
    print(f"  Properties Dir: {settings.NODE_PROPERTIES_DIR}")
    print(f"  Cache:    {settings.CACHE_DIR}")
    print("-" * 60)
    print("  Core Features:")
    print(f"    SSE Support:        {'Y' if HAS_SSE else 'N'}")
    print(f"    Anomaly Detection:  {'Y' if HAS_ANOMALY else 'N'}")
    print("-" * 60)
    print("  Renderer Configuration:")
    print(f"    Preference: {settings.RENDERER_PREFERENCE}")
    print(f"    cosmos.gl Min Nodes: {settings.RENDERER_COSMOS_MIN_NODES}")
    print(f"    cosmos.gl Preferred Nodes: {settings.RENDERER_COSMOS_PREFERRED_NODES}")
    print("-" * 60)
    print("  Layout Backends:")
    print(f"    Cytoscape Desktop:  {'Y' if HAS_CYTOSCAPE_DESKTOP else 'N'}")
    print(f"    igraph:             {'Y' if HAS_IGRAPH else 'N'}")
    print(f"    ForceAtlas2:        {'Y' if HAS_FA2 else 'N'}")
    print(f"    Local Spring:       Y")
    print(f"    Priority: {', '.join(settings.LAYOUT_BACKEND_PRIORITY)}")
    print("-" * 60)
    print("  Capacity Flow:")
    print(f"    OR-Tools:           {'Y' if HAS_ORTOOLS else 'N'}")
    print(f"    NetworkX:           Y")
    print(f"    Backend Priority: {', '.join(settings.CAPACITY_FLOW_BACKEND_PRIORITY)}")
    print("-" * 60)
    print("  Snapshots:")
    print(f"    Storage: {settings.SNAPSHOT_CACHE_DIR}")
    print(f"    SQL Templates: {settings.SNAPSHOT_SQL_DIR}")
    print(f"    Max Batch Size: {settings.SNAPSHOT_MAX_BATCH_SIZE}")
    print("-" * 60)
    print("  External API Properties:")
    print(f"    Base URL: {settings.EXTERNAL_API_BASE_URL}")
    print(f"    Providers: {', '.join(settings.EXTERNAL_API_PROVIDERS) or 'None'}")
    print(f"    Blacklist: {'Y' if settings.EXTERNAL_API_BLACKLIST_ENABLED else 'N'}")
    print("-" * 60)
    print("  UI Mode Configuration:")
    if settings.PRODUCTION_MODE:
        print("    Mode: Production")
        print(f"    Hidden Panels: {', '.join(settings.UI_HIDDEN_PANELS) or 'None'}")
        print(f"    Auto-Load: {'Y' if settings.AUTO_LOAD_ON_STARTUP else 'N'}")
        if settings.AUTO_LOAD_ON_STARTUP and settings.DEFAULT_SQL_FILES:
            print(f"    SQL Files: {', '.join(settings.DEFAULT_SQL_FILES)}")
            print(f"    Properties: {', '.join(settings.DEFAULT_PROPERTIES_FILES) or 'None'}")
            print(f"    Reload Interval: {settings.AUTO_RELOAD_DEFAULT_INTERVAL}s")
    else:
        print("    Mode: Admin (all panels visible, manual control)")
    print("=" * 60 + "\n")