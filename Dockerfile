FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyTorch Geometric
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads static/temp templates

EXPOSE 5000

CMD ["python", "app.py"]