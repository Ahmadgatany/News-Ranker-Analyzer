import os
import pandas as pd

# Define the data directory
RAW_DATA_DIR = "data/raw/"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Define the dataset path (assuming manual download)
DATA_FILE_PATH = os.path.join(RAW_DATA_DIR, "news_sentiment.csv")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(DATA_FILE_PATH)
print("Dataset loaded successfully!")

# Display basic dataset information
print("Dataset Overview:")
print(df.info())  # Show column names and data types
print(df.head())  # Show first 5 rows

# Initialize DVC for version control
print("Initializing DVC...")
os.system("dvc init")

# Add the dataset to DVC tracking
print("Adding dataset to DVC...")
os.system(f"dvc add {DATA_FILE_PATH}")

# Save DVC changes
print("Committing changes to DVC...")
os.system("git add . && git commit -m 'Added raw dataset with DVC tracking'")
print("Data collection and version control setup completed!")
