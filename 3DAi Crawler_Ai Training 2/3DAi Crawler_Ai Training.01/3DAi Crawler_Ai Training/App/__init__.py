# app/__init__.py
from flask import Flask
from celery import Celery

celery_app = None  # Initialize celery_app outside the function

def create_celery_app(app=None):
    """Factory function to create and configure the Celery app."""
    global celery_app
    celery_app = Celery(app.import_name, broker=app.config.get('CELERY_BROKER_URL', 'redis://redis:6379/0'), backend=app.config.get('CELERY_RESULT_BACKEND', 'redis://redis:6379/0'))
    celery_app.conf.update(app.config)
    return celery_app

def create_app():
    app = Flask(__name__)

    # Configurations (add your custom configurations here)
    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

    # Initialize Celery
    create_celery_app(app)

    return app