# Explaining Restaurant Ratings Using Customer Reviews
# Project Overview
Restaurants receive thousands of customer reviews, but star ratings alone do not explain why ratings increase or decline. Customer reviews contain rich qualitative information about food quality, service, pricing, ambience, and operational factors. However, this information is unstructured and difficult to analyze at scale, limiting restaurants’ ability to identify the true drivers of customer satisfaction and dissatisfaction. This project applies natural language processing (NLP) and machine learning to transform unstructured Yelp restaurant reviews into structured insights that explain rating behavior beyond star scores alone.

# Project Objectives
- Predict customer sentiment from review text using transformer-based models
- Identify key aspects (topics) discussed in restaurant reviews
- Link sentiment and aspects to restaurant operational attributes
- Explain why restaurant ratings vary using data-driven insights
- Provide an analytical framework for prioritizing operational improvements

# Methodology Overview
The project is structured into three logical notebooks, each representing a distinct stage of the analytics pipeline:

# Tools & Technologies
- Programming: Python
- Big Data Processing: PySpark
- NLP & ML: Transformers (DistilBERT), BERTopic, SentenceTransformers
- Libraries: Pandas, NumPy, Scikit-learn
- Visualization: Matplotlib, Seaborn
- Environment: Google Colab

# Notebook Overview
## Notebook 1: Model Development
Purpose: Sentiment Classification & Aspect Extraction Model Training
- Scalable data processing using PySpark
- Creation of sentiment labels from star ratings
- Fine-tuning a DistilBERT model for 3-class sentiment classification
- Training a BERTopic model for aspect (topic) extraction
- Evaluation using accuracy, precision, recall, and F1-score
- Trained models are saved for reuse
Outcome: Reusable sentiment and aspect extraction models
## Notebook 2: Sentiment Prediction & Aspect Extraction
Purpose: Apply trained models at scale
- Loads previously trained sentiment and topic models
- Performs batch sentiment prediction on restaurant reviews
- Extracts review-level aspects using BERTopic
- Combines model outputs with:
     - Business identifiers
     - Review text
     - Star ratings
     - Restaurant operational attributes
- Constructs a unified analytical dataset (aspect_df)
Outcome: Structured review-level dataset for insight analysis
# Notebook 3: Insight Analysis
Purpose: Explain rating behavior and identify drivers of sentiment
- Exploratory and diagnostic analysis on aspect_df
- Topic frequency and sentiment distribution analysis
- Net sentiment scoring by restaurant aspect
- Aspect-specific operational driver analysis
-  Pareto analysis of negative sentiment drivers
-   eatmaps for operational attribute diagnostics
Outcome: Actionable insights explaining why restaurant ratings vary

# Key Insights Enabled by This Project
- Identifies which restaurant aspects drive dissatisfaction despite high star ratings
- Reveals operational features that amplify or mitigate negative sentiment
- Applies Pareto analysis to prioritize high-impact improvement areas
- Bridges the gap between unstructured text data and business decision-making
