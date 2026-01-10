from models import sentiment_model, tokenizer, aspect_centroids, aspect_labels, embedding_model
import torch
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Sentiment prediction
# -------------------------
def predict_sentiment_single(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.cpu() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()

    return {
        "sentiment": ["negative", "neutral", "positive"][label],
        "confidence": float(probs.max())
    }

def predict_sentiment_batch(texts):
    return [predict_sentiment_single(text) for text in texts]

# -------------------------
# Aspect prediction
# -------------------------
def predict_aspect_centroid(text):
    emb = embedding_model.encode([text], convert_to_numpy=True)

    similarities = {
        topic_id: float(
            cosine_similarity(emb, centroid.reshape(1, -1))[0][0]
        )
        for topic_id, centroid in aspect_centroids.items()
    }

    best_topic = max(similarities, key=similarities.get)

    return {
        "aspect": aspect_labels[best_topic],
        "confidence": similarities[best_topic]
    }

def predict_aspect_centroid_batch(texts):
    return [predict_aspect_centroid(text) for text in texts]

# -------------------------
# Unified inference
# -------------------------
def inference_single(text):
    sentiment = predict_sentiment_single(text)
    aspect = predict_aspect_centroid(text)

    return {
        "review": text,
        "sentiment": sentiment["sentiment"],
        "sentiment_confidence": sentiment["confidence"],
        "aspect": aspect["aspect"],
        "aspect_confidence": aspect["confidence"]
    }

def inference_batch(texts):
    sentiment_results = predict_sentiment_batch(texts)
    aspect_results = predict_aspect_centroid_batch(texts)

    return [
        {
            "review": text,
            "sentiment": s["sentiment"],
            "sentiment_confidence": s["confidence"],
            "aspect": a["aspect"],
            "aspect_confidence": a["confidence"]
        }
        for text, s, a in zip(texts, sentiment_results, aspect_results)
    ]
