# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies (needed for PostgreSQL and numpy)
RUN apt-get update && apt-get install -y libpq-dev gcc

# Set the working directory to /app
WORKDIR /app

# Copy and install ROOT requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY graph_metrics.py .
COPY anomaly_engine.py .
COPY sql/ sql/
COPY web_viewer/ web_viewer/

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "web_viewer.app:app", "--host", "0.0.0.0", "--port", "8000"]