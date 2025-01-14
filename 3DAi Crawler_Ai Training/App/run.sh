#!/bin/bash

# Start the Celery worker
celery -A app.celery_app worker --loglevel=info &

# Optionally, start Flask for serving the web app
flask run --host=0.0.0.0 --port=5000