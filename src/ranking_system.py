import os
import pandas as pd

# Define paths
INPUT_FILE = "data/sentiment_results/logistic_regression_sentiment_results.csv"
OUTPUT_FILE = "data/ranking_results/ranked_news.csv"

# Ensure output directory exists
os.makedirs("data", exist_ok=True)

# Load sentiment results
df = pd.read_csv(INPUT_FILE)

# Ensure necessary columns exist
if "news" not in df.columns or "Logistic_Regression_Sentiment" not in df.columns:
    raise ValueError("The dataset must contain 'news' and 'Logistic_Regression_Sentiment' columns.")

# Convert sentiment to numerical values (if not already)
df["Sentiment_Score"] = df["Logistic_Regression_Sentiment"].map({1: 1, 0: 0})

# Rank news articles based on sentiment score
df_sorted = df.sort_values(by="Sentiment_Score", ascending=False)

# Save ranked news to CSV
df_sorted.to_csv(OUTPUT_FILE, index=False)
print(f"Ranked news articles saved to {OUTPUT_FILE}")
