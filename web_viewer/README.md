# Graph Analyzer - Network Visualization & Analysis Dashboard

A sophisticated web-based graph visualization and analysis dashboard built with FastAPI and Cytoscape.js. Designed for large-scale network analysis with support for millions of nodes and edges.

## Features

- **Interactive Graph Visualization**: WebGL-accelerated rendering with Cytoscape.js
- **Anomaly Detection**: Eight statistical and ML-based algorithms
- **Composite Metrics**: Create custom metrics from combinations of existing ones
- **Distribution Analysis**: Histograms, scatter plots, and PCA visualization
- **Auto-Reload**: Real-time data refresh with Server-Sent Events
- **Multiple Layout Algorithms**: Cytoscape Desktop integration, local spring layout, external services
- **Advanced Filtering**: Regex support, numeric/string/array filtering
- **Data Explorer**: Full tabular view with sorting, filtering, and export

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for layout service)
- PostgreSQL database with network data
- (Optional) Cytoscape Desktop for advanced layouts

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-repo/graph-analyzer.git
cd graph-analyzer
```

2. **Install Python dependencies**:
```bash
cd web_viewer
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

4. **Start the application**:
```bash
python run.py
```

5. **Open in browser**:
```
http://localhost:8000
```

### Docker Deployment

```bash
cd web_viewer
docker-compose up -d
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System architecture and components |
| [Algorithms](docs/ALGORITHMS.md) | Anomaly detection algorithms |
| [Graph Metrics](docs/METRICS.md) | 120+ computed graph metrics |
| [Features Guide](docs/FEATURES.md) | Complete feature documentation |
| [Filters & Search](docs/FILTERS.md) | Filtering with regex examples |
| [Composite Metrics](docs/COMPOSITE_METRICS.md) | Creating custom metrics |
| [API Reference](docs/API.md) | REST API documentation |
| [Configuration](docs/CONFIGURATION.md) | Environment variables |
| [Deployment](docs/DEPLOYMENT.md) | Production deployment guide |

## Project Structure

```
graph-analyzer/
├── web_viewer/                # Main web application
│   ├── backend/               # FastAPI backend
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Business logic
│   │   ├── models/            # Pydantic models
│   │   └── utils/             # Helper functions
│   ├── engines/               # Computation engines
│   │   ├── algorithms/        # Anomaly detection algorithms
│   │   ├── anomaly_engine.py  # Main anomaly orchestrator
│   │   ├── composite_engine.py# Composite metrics
│   │   └── graph_metrics.py   # NetworkX metrics
│   ├── static/                # Frontend assets
│   │   ├── js/                # JavaScript modules
│   │   ├── css/               # Stylesheets
│   │   └── *.html             # HTML pages
│   ├── layout_service/        # Node.js layout service
│   └── cache/                 # Cached data and layouts
├── sql/                       # SQL query files
│   └── properties/            # Node properties SQL
├── cytoscape/                 # Cytoscape Desktop config
└── figs/                      # Example visualizations
```

## Screenshots

### Main Dashboard
![Dashboard](img/cytoscape_webapp.png)

### Cytoscape Desktop Integration
![Cytoscape Desktop](img/cytoscape_desktop.png)

## Technology Stack

- **Backend**: FastAPI, Python 3.10+, NetworkX, scikit-learn
- **Frontend**: Vanilla JavaScript, Cytoscape.js, Chart.js
- **Database**: PostgreSQL with SQLAlchemy
- **Real-time**: Server-Sent Events (SSE)
- **Layout**: Cytoscape Desktop (py4cytoscape), Node.js service

## License

MIT License - see [LICENSE](LICENSE) for details.
