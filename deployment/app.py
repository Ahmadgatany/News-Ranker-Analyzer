from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define FastAPI app
app = FastAPI()

# Define paths to model and tokenizer
MODEL_PATH = "../models/logistic_regression_model.pkl"
VECTORIZER_PATH = "../models/tfidf_vectorizer.pkl"

# Load the TF-IDF vectorizer
print("Loading TF-IDF vectorizer...")
with open(VECTORIZER_PATH, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load Logistic Regression model
print("Loading Logistic Regression model...")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Define request body
class NewsInput(BaseModel):
    text: str

# Define prediction function
def predict_sentiment(text: str):
    """Predicts sentiment as either NEGATIVE or POSITIVE using Logistic Regression and TF-IDF."""
    # Transform the input text using the TF-IDF vectorizer
    input_features = tfidf_vectorizer.transform([text])
    
    # Make prediction with the logistic regression model
    prediction = model.predict(input_features)
    
    # Map the prediction to sentiment
    sentiment = "NEGATIVE" if prediction[0] == 0 else "POSITIVE"
    
    return sentiment

# Define API endpoint
@app.post("/predict")
def predict(news: NewsInput):
    """API endpoint to receive a news article and return its sentiment."""
    sentiment = predict_sentiment(news.text)
    return {"text": news.text, "sentiment": sentiment}

# Root endpoint
@app.get("/")
def home():
    """Root endpoint to check if the API is running."""
    return {"message": "Logistic Regression Sentiment Analysis API is running!"}
