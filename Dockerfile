FROM python:3.14-slim

# Prevent interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: Tesseract + Portuguese language pack + OpenCV/ONNX runtime C libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-por \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/   ./src/
COPY tests/ ./tests/
COPY requirements.txt requirements-dev.txt Makefile README.md ./

# Create runtime directories
RUN mkdir -p data labels/back

VOLUME ["/app/data", "/app/labels"]

# Labeling server default port (--port in server.py)
EXPOSE 8765

# Default: run the full processing pipeline.
CMD ["python", "src/pipeline/data_pipeline.py"]
