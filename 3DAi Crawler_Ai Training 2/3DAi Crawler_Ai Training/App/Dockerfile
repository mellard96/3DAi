# Use Python base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /App

# Copy requirements file into the container
COPY requirements.txt .

# Install system dependencies for building torch and NumPy compatibility
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using a mirror and increased timeout
RUN pip install --timeout=1000 --no-cache-dir -i https://pypi.doubanio.com/simple/ -r requirements.txt

# Install Spacy and download the en_core_web_sm model
RUN python -m spacy download en_core_web_sm

# Copy the application code into the container
COPY . /app/

# Expose the port for Flask (optional)
EXPOSE 5000

# Command to run Celery worker by default
CMD ["celery", "-A", "app.celery_app", "worker", "--loglevel=info"]
