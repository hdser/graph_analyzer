# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (needed for PostgreSQL and numpy)
RUN apt-get update && apt-get install -y libpq-dev gcc

# Set the working directory to /app (this will represent your project root)
WORKDIR /app

# 1. Copy and install ROOT requirements (for graph_metrics.py)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy and install WEB_VIEWER requirements (for app.py)
COPY web_viewer/requirements.txt web_viewer/
RUN pip install --no-cache-dir -r web_viewer/requirements.txt

# 3. Copy the necessary source code
# We copy specific files to keep the image clean
COPY graph_metrics.py .
COPY sql/ sql/
COPY web_viewer/ web_viewer/

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the application from the ROOT directory so imports work
# We use the module path "web_viewer.app:app"
CMD ["uvicorn", "web_viewer.app:app", "--host", "0.0.0.0", "--port", "8000"]