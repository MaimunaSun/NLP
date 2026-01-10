import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# -------------------------
# Paths to saved models
# -------------------------
SENTIMENT_MODEL_DIR = "restaurant_sentiment_model"
ASPECT_CENTROIDS_PATH = "restaurant_sentiment_model/aspect_centroids.pkl"

# -------------------------
# Load sentiment model
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_DIR)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR)
sentiment_model.eval()

# -------------------------
# Load aspect centroids
# -------------------------
with open(ASPECT_CENTROIDS_PATH, "rb") as f:
    aspect_data = pickle.load(f)

aspect_centroids = aspect_data["centroids"]  # {topic_id: centroid vector}
aspect_labels = aspect_data["labels"]        # {topic_id: label string}

# -------------------------
# Embedding model for new text
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
