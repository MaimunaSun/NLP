# NLP-Restaurant-Review-Insights

## Project Overview
Restaurants receive thousands of customer reviews, but star ratings alone do not explain why ratings increase or decline. Customer reviews contain rich qualitative information about food quality, service, pricing, ambience, and operational factors. However, this information is unstructured and difficult to analyze at scale, limiting restaurants’ ability to identify the true drivers of customer satisfaction and dissatisfaction.  

This project uses aspect-based sentiment analysis to transform unstructured Yelp restaurant reviews into structured insights that explain rating behavior beyond star scores alone.

The **ML_Production** folder contains production-ready code, pre-trained models, and Docker setup for deploying sentiment and aspect extraction APIs.

---

## Project Objectives
- Predict customer sentiment from review text using transformer-based models  
- Identify key aspects (topics) discussed in restaurant reviews  
- Link sentiment and aspects to restaurant operational attributes  
- Explain why restaurant ratings vary using data-driven insights  
- Deploy models in production with a REST API and Docker  

---
## Tools & Technologies
- **Programming:** Python  
- **Big Data Processing:** PySpark  
- **NLP & ML:** Transformers (DistilBERT), BERTopic, SentenceTransformers  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Deployment:** FastAPI, Docker  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Google Colab for development notebooks
- 
## Project Structure
---

## Notebook Overview

### Notebook 1: Model Development
**Purpose:** Train sentiment and aspect extraction models  
- Data processing using PySpark  
- Create sentiment labels from star ratings  
- Fine-tune DistilBERT for 3-class sentiment classification  
- Train BERTopic for aspect extraction  
- Evaluate models with accuracy, precision, recall, and F1-score  
- Save trained models for production deployment  

**Outcome:** Reusable sentiment and aspect extraction models  

### Notebook 2: Sentiment Prediction & Aspect Extraction
**Purpose:** Apply trained models at scale  
- Load previously trained sentiment and topic models  
- Perform batch sentiment prediction on restaurant reviews  
- Extract review-level aspects using BERTopic  
- Combine outputs with business IDs, review text, star ratings, and operational attributes  
- Construct a unified analytical dataset (`aspect_df`)  

**Outcome:** Structured review-level dataset for insight analysis  

### Notebook 3: Insight Analysis
**Purpose:** Explain rating behavior and identify drivers of sentiment  
- Exploratory and diagnostic analysis on `aspect_df`  
- Topic frequency and sentiment distribution analysis  
- Net sentiment scoring by restaurant aspect  
- Aspect-specific operational driver analysis  
- Pareto analysis of negative sentiment drivers  
- Heatmaps for operational attribute diagnostics  

**Outcome:** Actionable insights explaining why restaurant ratings vary  

---

## ML_Production Folder
**Purpose:** Production-ready sentiment and aspect extraction API.  

**Contents:**  
- `app.py` – FastAPI application serving inference endpoints (`/predict` and `/predict_batch`)  
- `models.py` – Loads pre-trained sentiment and aspect models  
- `inference.py` – Handles single and batch predictions  
- `requirements.txt` – Python dependencies  
- `Dockerfile` & `.dockerignore` – Containerization setup  
- `restaurant_sentiment_model/` – Pre-trained models (mounted as a Docker volume, excluded from Git for size)
  
## Key Insights Enabled by This Project
- Identifies which restaurant aspects drive dissatisfaction despite high star ratings  
- Reveals operational features that amplify or mitigate negative sentiment  
- Applies Pareto analysis to prioritize high-impact improvement areas  
- Bridges the gap between unstructured text data and business decision-making  
- Enables production-ready deployment for real-time review analysis via **ML_Production**  

---

## Future Work
Based on the current implementation, the following extensions could further enhance the project:

- **Interactive Dashboard Integration:** Use outputs from the ML_Production API to build dashboards in Streamlit, Tableau, or Power BI, allowing stakeholders to monitor sentiment trends and key drivers over time.  
- **Streaming Data Support:** Integrate with streaming pipelines (e.g., Kafka) to process incoming reviews in near real-time and automatically update sentiment and aspect scores.  
- **Domain-Specific Model Enhancements:** Fine-tune models for specific cuisines or regional restaurant types to improve aspect recognition and sentiment accuracy. 
- **Automated Model Retraining:** Implement scheduled retraining workflows to continuously improve model performance on new review data.  
