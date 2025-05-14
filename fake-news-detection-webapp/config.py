import os
from pathlib import Path

# Base paths
BASE_DIR = "fake-news-nlp/saved_models/"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Model configuration
DEFAULT_MODEL = "bert"  # Options: bert, distilbert, roberta
MODEL_PATH = os.path.join(MODEL_DIR, "fnc1_bert_weighted_best.pt")  # Update with your model path
TOKENIZER_PATH = "bert-base-uncased"  # Default tokenizer
MAX_LENGTH = 512  # Maximum sequence length

# Available models
MODELS = {
    "bert": {
        "name": "bert-base-uncased",
        "type": "bert",
        "path": os.path.join(MODEL_DIR, "fnc1_bert_weighted_best.pt"),
    },
    "distilbert": {
        "name": "distilbert-base-uncased",
        "type": "distilbert",
        "path": os.path.join(MODEL_DIR, "fnc1_distilbert_weighted_best.pt"),
    },
    "roberta": {
        "name": "roberta-base",
        "type": "roberta",
        "path": os.path.join(MODEL_DIR, "fnc1_roberta_weighted_best.pt"),
    }
}

# API Configuration
API_V1_PREFIX = "/api/v1"

# Labels (keep consistent with your trained model)
LABEL_MAPPING = {
    0: "agree",
    1: "disagree",
    2: "discuss",
    3: "unrelated"
}

# Web app configuration
DEBUG = True
APP_NAME = "Fake News Detector"
APP_DESCRIPTION = "A web application for detecting fake news using NLP models"