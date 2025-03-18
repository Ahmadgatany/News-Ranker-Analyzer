import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Paths for raw and processed data
RAW_DATA_PATH = "data/raw/news_sentiment.csv"
PROCESSED_DATA_PATH = "data/processed/news_sentiment_cleaned.csv"

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(RAW_DATA_PATH)
print("Dataset loaded successfully!")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    """Remove punctuation, numbers, special characters, and convert text to lowercase."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters and punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to preprocess text
def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Apply preprocessing to news column
print("Cleaning and preprocessing news data...")
df["cleaned_news"] = df["news"].apply(lambda x: preprocess_text(clean_text(x)))
print("Preprocessing completed!")

# Save the processed dataset
os.makedirs("data/processed", exist_ok=True)  # Ensure processed data folder exists
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Processed data saved to {PROCESSED_DATA_PATH}")
