#!/bin/bash

# Start the Celery worker in the background
celery -A app.celery_app worker --loglevel=info --uid=1000 --gid=1000 &

# Start the Flask app in the foreground
flask run --host=0.0.0.0 --port=5000