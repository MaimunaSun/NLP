from fastapi import FastAPI
from pydantic import BaseModel
from inference import inference_single, inference_batch

app = FastAPI(title="Restaurant Review Analyzer")

# -------------------------
# Request models
# -------------------------
class Review(BaseModel):
    text: str

class BatchReviews(BaseModel):
    reviews: list[str]

# -------------------------
# Endpoints
# -------------------------
@app.get("/")
def root():
    return {"message": "Restaurant Review Analyzer API"}

@app.post("/predict")
def predict(review: Review):
    return inference_single(review.text)

@app.post("/predict_batch")
def predict_batch(batch: BatchReviews):
    return inference_batch(batch.reviews)
