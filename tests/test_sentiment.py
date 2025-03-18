import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths for model, vectorizer, and test dataset
MODEL_PKL_PATH = "models/logistic_regression_model.pkl"
VECTORIZER_PKL_PATH = "models/tfidf_vectorizer.pkl"
TEST_FILE = "tests/test_news.csv"
OUTPUT_FILE = "tests/new_predictions.csv"


# Load the test dataset
print(f"Loading new dataset from {TEST_FILE}...")
df_test = pd.read_csv(TEST_FILE)

# Ensure the necessary column exists
if "news" not in df_test.columns:
    raise ValueError("The test dataset must contain a 'news' column for sentiment analysis.")

# Load the trained model and vectorizer
print("Loading trained model and vectorizer...")
with open(MODEL_PKL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PKL_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Convert text data to TF-IDF features
X_test = vectorizer.transform(df_test["news"])

# Predict sentiment
print("Predicting sentiment for new data...")
df_test["Predicted_Sentiment"] = model.predict(X_test)

# Convert numerical predictions to labels
df_test["Predicted_Sentiment"] = df_test["Predicted_Sentiment"].map({1: "Positive", 0: "Negative"})

# Save results to a new CSV file
print(f"Saving sentiment predictions to {OUTPUT_FILE}...")
df_test.to_csv(OUTPUT_FILE, index=False)

print("Sentiment analysis for new data completed successfully!")


