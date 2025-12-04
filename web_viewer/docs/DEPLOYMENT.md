# Deployment Guide

This guide covers deploying Graph Analyzer to production environments.

## Deployment Options

| Option | Best For | Complexity |
|--------|----------|------------|
| Local Development | Testing, development | Low |
| Docker Compose | Single server | Medium |
| Kubernetes | Scalability, high availability | High |
| Digital Ocean App Platform | Managed deployment | Low |

---

## Local Development

### Prerequisites

- Python 3.10+
- Node.js 18+ (for layout service)
- PostgreSQL database
- (Optional) Cytoscape Desktop

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/graph-analyzer.git
cd graph-analyzer/web_viewer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start application
python run.py
```

### Start Layout Service (Optional)

```bash
cd layout_service
npm install
npm start
```

### Access

- Application: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Layout Service: http://localhost:3000

---

## Docker Deployment

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create cache directories
RUN mkdir -p cache/layouts cache/data

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "run.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - DB_NAME=circles
      - DB_USER=readonly_user
      - DB_PASSWORD=${DB_PASSWORD}
      - LAYOUT_SERVICE_URL=http://layout:3000/layout
      - HIDE_DATA_SOURCE_UI=true
      - DEFAULT_SQL_FILES=crc_v2_trusts.sql
      - DEFAULT_METRICS_MODE=essential
    volumes:
      - ./sql:/app/sql:ro
      - cache_data:/app/cache
    depends_on:
      - db
      - layout
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  layout:
    build: ./layout_service
    ports:
      - "3000:3000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=circles
      - POSTGRES_USER=readonly_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
  cache_data:
```

### Build and Run

```bash
# Create .env file
echo "DB_PASSWORD=your-secure-password" > .env

# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop services
docker-compose down
```

---

## Production Configuration

### Environment Variables

```bash
# .env.production
# Database (read-only user recommended)
DB_HOST=your-db-host.com
DB_PORT=5432
DB_NAME=production_db
DB_USER=readonly_user
DB_PASSWORD=strong-password-here

# Layout
LAYOUT_SERVICE_URL=http://layout:3000/layout

# UI Mode
HIDE_DATA_SOURCE_UI=true
DEFAULT_SQL_FILES=production_trusts.sql,production_flows.sql
DEFAULT_PROPERTIES_FILES=production_avatars.sql
DEFAULT_METRICS_MODE=essential

# Performance
N_JOBS=4
EDGE_CHUNK_SIZE=100000

# Auto-reload (5 minutes)
AUTO_RELOAD_DEFAULT_INTERVAL=300
```

### Security Considerations

1. **Database User**: Use read-only database user
2. **Environment Variables**: Never commit secrets to git
3. **HTTPS**: Use reverse proxy with TLS
4. **CORS**: Restrict to known origins
5. **Rate Limiting**: Implement at load balancer

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/graph-analyzer
server {
    listen 80;
    server_name graph.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name graph.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/graph.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/graph.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
    }

    # Large file uploads for data
    client_max_body_size 100M;
}
```

---

## Digital Ocean Deployment

### App Platform

1. **Create App** from GitHub repository
2. **Configure Build**:
   - Build Command: `pip install -r web_viewer/requirements.txt`
   - Run Command: `cd web_viewer && python run.py`
3. **Set Environment Variables** in App Settings
4. **Add Database** component (managed PostgreSQL)
5. **Deploy**

### Droplet Deployment

```bash
# SSH to droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin

# Clone repository
git clone https://github.com/your-repo/graph-analyzer.git
cd graph-analyzer/web_viewer

# Configure and start
cp .env.example .env
nano .env  # Edit settings
docker compose up -d
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-analyzer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: graph-analyzer
  template:
    metadata:
      labels:
        app: graph-analyzer
    spec:
      containers:
      - name: web
        image: your-registry/graph-analyzer:latest
        ports:
        - containerPort: 8000
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: host
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: cache-volume
        persistentVolumeClaim:
          claimName: graph-analyzer-cache
---
apiVersion: v1
kind: Service
metadata:
  name: graph-analyzer
spec:
  selector:
    app: graph-analyzer
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Persistent Volume Claim

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: graph-analyzer-cache
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

---

## Monitoring

### Health Checks

The `/health` endpoint returns:

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "mode": "production",
  "data_status": "ready",
  "graphs_loaded": true,
  "node_count": 50000
}
```

### Metrics to Monitor

| Metric | Warning | Critical |
|--------|---------|----------|
| Response Time | > 5s | > 30s |
| Error Rate | > 1% | > 5% |
| Memory Usage | > 70% | > 90% |
| CPU Usage | > 70% | > 90% |
| Disk Usage | > 70% | > 90% |

### Logging

Logs are written to stdout and can be collected with:
- Docker: `docker logs` or logging driver
- Kubernetes: fluentd, Loki, etc.
- Standalone: redirect to file or syslog

---

## Scaling

### Horizontal Scaling

For multiple instances:
1. Use shared database (PostgreSQL)
2. Use shared cache (Redis or NFS for Parquet files)
3. Load balance with sticky sessions (for SSE)

### Vertical Scaling

| Users | Recommended Resources |
|-------|----------------------|
| 1-5 | 2 CPU, 4GB RAM |
| 5-20 | 4 CPU, 8GB RAM |
| 20-50 | 8 CPU, 16GB RAM |
| 50+ | Consider horizontal scaling |

### Cache Optimization

For large datasets:
```bash
# Pre-compute layouts
python -c "from backend.services.layout_service import LayoutService; ..."

# Pre-compute metrics
python -c "from engines.graph_metrics import GraphMetrics; ..."
```

---

## Backup and Recovery

### Data to Backup

1. **PostgreSQL Database**: Contains source network data
2. **Cache Directory**: Layouts and computed metrics
3. **Configuration**: .env files (encrypted)

### Backup Commands

```bash
# Database backup
pg_dump -h localhost -U user -d circles > backup.sql

# Cache backup
tar -czf cache-backup.tar.gz cache/

# Restore database
psql -h localhost -U user -d circles < backup.sql

# Restore cache
tar -xzf cache-backup.tar.gz
```

---

## Troubleshooting

### Application Won't Start

```bash
# Check logs
docker logs graph-analyzer-web-1

# Common issues:
# - Database connection failed: check DB_HOST, DB_PORT
# - Module not found: rebuild image
# - Port in use: change port mapping
```

### Slow Performance

1. Enable performance mode in UI
2. Reduce metrics mode (use "basic")
3. Increase container resources
4. Check database query performance

### Memory Issues

```bash
# Increase memory limit
docker update --memory 4g graph-analyzer-web-1

# Or in docker-compose.yml:
services:
  web:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Database Connection Issues

```bash
# Test connection
docker exec -it graph-analyzer-web-1 python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:pass@host:5432/db')
conn = engine.connect()
print('Connected!')
"
```

---

## Updates

### Rolling Update (Docker Compose)

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build web
docker-compose up -d --no-deps web
```

### Blue-Green Deployment

```bash
# Start new version on different port
docker-compose -f docker-compose.blue.yml up -d

# Test new version
curl http://localhost:8001/health

# Switch traffic (update nginx/load balancer)

# Stop old version
docker-compose -f docker-compose.green.yml down
```