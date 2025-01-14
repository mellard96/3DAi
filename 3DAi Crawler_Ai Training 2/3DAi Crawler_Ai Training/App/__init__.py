from flask import Flask
from celery import Celery

def create_celery_app(app=None):
    """Factory function to create and configure the Celery app."""
    celery = Celery(app.import_name, broker=app.config.get('CELERY_BROKER_URL', 'redis://redis:6379/0'))
    celery.conf.update(app.config)
    return celery

def create_app():
    app = Flask(__name__)

    # Configurations (add your custom configurations here)
    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'

    # Initialize Celery
    global celery_app
    celery_app = create_celery_app(app)

    return app

# Create the Flask app instance
app = create_app()

# Note: No import statement for tasks here
# You can initialize or trigger tasks as needed in `app.py`
