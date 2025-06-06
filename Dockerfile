FROM python:3.10-slim-bookworm
EXPOSE 8005

ENV PATH="${PATH}:/root/.local/bin" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libffi-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

WORKDIR /app

# Copy only dependency files
COPY pyproject.toml poetry.lock ./

# Install poetry and production dependencies
RUN python3 -m pip install --no-cache-dir pipx && \
    python3 -m pipx ensurepath --global && \
    python3 -m pipx install poetry && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi --only main && \
    rm -rf ~/.cache/pip ~/.cache/pipx

# Copy only necessary application files
COPY main.py model.py ./
RUN mkdir -p ./models
COPY ./models/model.onnx ./models/

# Run the FastAPI application
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
