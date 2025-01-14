import os
import requests
from bs4 import BeautifulSoup
from celery import Celery
import json
import time
import re
import spacy
from transformers import pipeline
from flask import Flask
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Celery configuration
def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['result_backend'], broker=app.config['broker_url'])
    celery.conf.update(app.config)
    return celery

# Function to create and configure the Flask app
def create_app():
    app = Flask(__name__)

    # Configurations
    app.config.update(
        broker_url='redis://redis:6379/0',
        result_backend='redis://redis:6379/0'
    )

    return app

# Create the Flask app instance
app = create_app()

# Create the Celery app instance
celery_app = make_celery(app)

# List of recommended sites to scrape for 3D model-related data
RECOMMENDED_SITES = [
    "https://www.thingiverse.com",
    "https://www.myminifactory.com",
    "https://www.grabcad.com",
    "https://www.yeggi.com",
    "https://cults3d.com",
    "https://www.shapeways.com",
    "https://www.cgtrader.com",
    "https://polyhaven.com",
    "https://www.turbosquid.com",
    "https://www.3dwarehouse.sketchup.com",
    "https://all3dp.com",
    "https://www.tinkercad.com",
    "https://www.printables.com",
    "https://github.com",
    "https://www.openscad.org",
    "https://librecad.org",
    "https://www.thangs.com",
    "https://github.com/SoftFever/OrcaSlicer",
    "https://github.com",
    "https://www.open3d.org"
]

# Directory to store scraped and pruned data
DATA_STORAGE_PATH = "/app/data"
os.makedirs(DATA_STORAGE_PATH, exist_ok=True)

# Load spaCy NLP model for text processing
nlp = spacy.load("en_core_web_sm")

# Initialize Huggingface transformer pipeline for text summarization or other NLP tasks
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Web Crawler Task
@celery_app.task(name="web_crawler_task")
def web_crawler_task(start_urls, depth=1):
    """Crawl web pages starting from a list of URLs, specifically targeting 3D model-related resources."""
    crawled_data = []

    def crawl(url, current_depth):
        if current_depth > depth:
            return
        try:
            time.sleep(1)  # Add delay to avoid overwhelming servers
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract 3D model-related data and links
            keywords = re.compile(r'\b(3d model|3d printing|slicing|textures|fuzzyskin|beta modeling|experimental slicing|ai-powered 3d modeling|file conversion|minimal support methods|string art|prompt-based 3d model generation|3d editing|open source|repositories|3d capabilities|3d textures|gcode optimization|ai gcode|file conversion between common formats)\b', re.IGNORECASE)
            if keywords.search(soup.text):
                crawled_data.append({
                    'url': url,
                    'title': soup.title.string if soup.title else 'No Title',
                    'text': soup.get_text(strip=True)
                })

            for link in soup.find_all('a', href=True):
                if link['href'].startswith('http'):
                    crawl(link['href'], current_depth + 1)

        except requests.exceptions.RequestException as e:
            logger.error(f"Error crawling {url}: {e}")
            crawled_data.append({
                'url': url,
                'error': str(e)
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred while crawling {url}: {e}")
            crawled_data.append({
                'url': url,
                'error': str(e)
            })


    for url in start_urls:
        crawl(url, 0)

    # Save crawled data locally
    file_path = os.path.join(DATA_STORAGE_PATH, f"crawled_data_{start_urls[0].replace('https://', '').replace('/', '_')}.json")
    with open(file_path, 'w') as f:
        json.dump(crawled_data, f, indent=4)

    return file_path

# AI Model Training Task
@celery_app.task(name="train_model_task")
def train_model_task(data_file_path):
    """Train an AI model using the crawled 3D model-related data."""
    logger.info(f"Starting training task with data from {data_file_path}")

    if not os.path.exists(data_file_path):
        logger.error(f"Data file {data_file_path} does not exist.")
        return f"Data file {data_file_path} does not exist."

    with open(data_file_path, 'r') as f:
        crawled_data = json.load(f)

    # Data pruning: Keep entries with sufficient content
    processed_data = [entry for entry in crawled_data if len(entry.get('text', '')) > 100]

    if not processed_data:
        logger.warning("No data available after pruning.")
        return "No data available after pruning."

    # Data labeling (simplified example)
    labels = []
    for entry in processed_data:
        text = entry['text'].lower()
        if "printing" in text:
            labels.append("printing")
        elif "modeling" in text:
            labels.append("modeling")
        else:
            labels.append("other")

    # Filter entries without valid labels
    processed_data_with_labels = [{"text": entry["text"], "label": label} for entry, label in zip(processed_data, labels) if label != "other"]
    
    if not processed_data_with_labels:
        return "No data with valid labels found for training."

    texts = [entry["text"] for entry in processed_data_with_labels]
    labels = [entry["label"] for entry in processed_data_with_labels]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Create a pipeline with TF-IDF vectorizer and logistic regression
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression()),
        ]
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model training completed with accuracy: {accuracy}")

    # Save the trained model
    model_file_path = os.path.join(DATA_STORAGE_PATH, "trained_model.joblib")
    joblib.dump(model, model_file_path)
    logger.info(f"Trained model saved to {model_file_path}")

    # Save pruned data for model training
    pruned_file_path = data_file_path.replace("crawled_data", "pruned_data")
    with open(pruned_file_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    model_output = {
        "num_samples": len(processed_data),
        "accuracy": accuracy,
        "status": "Training complete",
        "model_file": model_file_path,
        "sample_data": processed_data[:1],
    }

    return model_output

# Automatically trigger crawling and training
if __name__ == "__main__":
    start_urls = ["https://www.thangs.com", "https://github.com/SoftFever/OrcaSlicer", "https://github.com"]
    result = web_crawler_task.delay(start_urls=start_urls, depth=1)
    result.then(lambda res: train_model_task.delay(res.get()))