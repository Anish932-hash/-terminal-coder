# Terminal Coder - Modern Multi-stage Dockerfile
# Optimized for Python 3.13 and modern containerization practices

# Use Python 3.13 slim image as base
FROM python:3.13-slim-bookworm as base

# Set environment variables for Python optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group --uid 1000 tcoder

# Build stage
FROM base as builder

# Install build dependencies
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# Production stage
FROM base as production

# Copy Python packages from builder
COPY --from=builder /root/.local /home/tcoder/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=tcoder:tcoder . .

# Install the package
RUN pip install --user --no-deps -e .

# Switch to non-root user
USER tcoder

# Add local bin to PATH
ENV PATH="/home/tcoder/.local/bin:$PATH"

# Set up application directories
RUN mkdir -p /home/tcoder/.terminal_coder/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import terminal_coder; print('OK')" || exit 1

# Default command
CMD ["terminal-coder", "--help"]

# Development stage
FROM production as development

# Switch back to root to install dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tmux \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    mypy \
    ruff \
    pre-commit

# Switch back to tcoder user
USER tcoder

# Set up development environment
RUN git config --global user.name "Terminal Coder Dev" \
    && git config --global user.email "dev@terminalcoder.local"

# Override default command for development
CMD ["bash"]

# Labels for metadata
LABEL org.opencontainers.image.title="Terminal Coder" \
      org.opencontainers.image.description="Advanced AI-Powered Development Terminal" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="Terminal Coder Team" \
      org.opencontainers.image.url="https://github.com/terminalcoder/terminal_coder" \
      org.opencontainers.image.source="https://github.com/terminalcoder/terminal_coder" \
      org.opencontainers.image.documentation="https://docs.terminalcoder.dev" \
      org.opencontainers.image.licenses="MIT"