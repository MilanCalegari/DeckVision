FROM python:3.10-slim

WORKDIR /app
ARG HF_TOKEN
ENV HUGGINGFACE_TOKEN=$HF_TOKEN

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    unzip \
    procps

# Copy project files
COPY requirements.txt .
COPY src/ src/
COPY scripts/ scripts/
COPY config.yaml .

# Install Python dependencies
RUN pip install -r requirements.txt

# Add src directory to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Create cache directory with proper permissions
RUN mkdir -p /.cache && \
    chmod 777 /.cache && \
    mkdir -p /app/db && \
    chmod 777 /app/db

# Create database directory
RUN mkdir -p /app/db

# Create data directory and download dataset
RUN mkdir -p /app/data && \
    curl -L -o /app/tarot-json.zip \
    https://www.kaggle.com/api/v1/datasets/download/lsind18/tarot-json && \
    cd /app && \
    unzip -o tarot-json.zip -d data && \
    rm tarot-json.zip

# Streamlit port
EXPOSE 8501

RUN python scripts/login_huggingface.py

# Run initialization scripts and app
CMD python scripts/create_and_write_features.py && \
    streamlit run scripts/streamlit_app.py