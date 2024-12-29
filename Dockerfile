    FROM python:3.10-slim

    WORKDIR /app

    # Install system dependencies with cache
    RUN --mount=type=cache,target=/var/cache/apt \
        apt-get update && apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
        unzip \
        procps

    # Install Ollama
    RUN curl -fsSL https://ollama.ai/install.sh | sh

    # Copy project files
    COPY requirements.txt .
    COPY src/ src/
    COPY scripts/ scripts/
    COPY config.yaml .

    # Install Python dependencies with cache
    RUN --mount=type=cache,target=/root/.cache/pip \
        pip install -r requirements.txt

    # Add src directory to PYTHONPATH
    ENV PYTHONPATH="${PYTHONPATH}:/app"

    # Create database directory
    RUN mkdir -p /app/db

    # Expose Streamlit and Ollama ports
    EXPOSE 8501 11434

    # Download and extract data with cache
    RUN target=/app/data \
        curl -L -o /app/tarot-json.zip \
        https://www.kaggle.com/api/v1/datasets/download/lsind18/tarot-json && \
        mkdir -p /app/data && \
        cd /app && \
        unzip -o tarot-json.zip -d data && \
        ls -la /app/data

    # Download LLM model with cache
    RUN --mount=type=cache,target=/root/.ollama \
        mkdir -p /root/.ollama && \
        ollama serve & \
        sleep 5 && \
        ollama pull llama3.2

    # Command to run create_and_write_features first, then Ollama and the app
    CMD python scripts/create_and_write_features.py && \
        ollama serve & python -m streamlit run scripts/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.enableCORS=false