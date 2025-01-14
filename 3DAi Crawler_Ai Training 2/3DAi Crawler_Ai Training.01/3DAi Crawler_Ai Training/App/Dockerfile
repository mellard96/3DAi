# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch with GPU support
RUN pip install --timeout=1000 --no-cache-dir -i https://pypi.doubanio.com/simple/ torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --find-links https://download.pytorch.org/whl/torch_stable.html
# Install other dependencies
RUN pip install --timeout=1000 --no-cache-dir -i https://pypi.doubanio.com/simple/ -r requirements.txt

# Install spaCy and download the en_core_web_sm model
RUN python -m spacy download en_core_web_sm

# Copy only the app directory
COPY . .

# Create a non-root user
RUN groupadd -r myuser && useradd -r -g myuser myuser

# Change ownership of the app directory to the non-root user
RUN chown -R myuser:myuser /app

# Switch to the non-root user
USER myuser

# Expose the port for Flask (optional)
EXPOSE 5000

# Command to run Celery worker as non-root user
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info", "--uid=1000", "--gid=1000"]