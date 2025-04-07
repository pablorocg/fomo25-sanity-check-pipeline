FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Ensures output is not buffered
ENV PYTHONUNBUFFERED=1
ENV LC_ALL=C.UTF-8

# Create necessary directories
RUN mkdir -p /input /output /results /app

WORKDIR /app

# Copy application files 
COPY src/ /app/

# Install dependencies as root
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install -e /app

# Create non-root user and change ownership of relevant directories
RUN groupadd -r user && useradd -m --no-log-init -r -g user user && \
    chown -R user:user /input /output /results /app

# Switch to non-root user for runtime
USER user

# Make scripts executable
RUN chmod +x /app/*.py

# Define entrypoint
ENTRYPOINT ["python", "/app/predict.py", "--input", "/input", "--output", "/output"]
