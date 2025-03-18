import os
import yaml
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load configuration from config.yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Extract paths and parameters from the config file
experiment_name = config["mlflow"]["experiment_name"]
tracking_uri = config["mlflow"]["tracking_uri"]
input_file = config["data_paths"]["processed"]
output_file = config["data_paths"]["results"]
model_pkl_path = config["output_paths"]["model"]
vectorizer_pkl_path = config["output_paths"]["vectorizer"]
model_name = config["model_params"]["model_name"]
max_iter = config["model_params"]["max_iter"]
vectorizer_max_features = config["model_params"]["vectorizer_max_features"]

# Ensure required directories exist
os.makedirs(os.path.dirname(output_file), exist_ok=True)
os.makedirs(os.path.dirname(model_pkl_path), exist_ok=True)

# Set up MLflow tracking
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# Load the processed dataset
print(f"Loading dataset from {input_file}...")
df = pd.read_csv(input_file)

# Ensure the necessary columns exist
if "news" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("The dataset must contain 'news' and 'sentiment' columns.")

# Convert sentiment labels to numerical values (Positive -> 1, Negative -> 0)
df['sentiment'] = df['sentiment'].map({"POSITIVE": 1, "NEGATIVE": 0})

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=vectorizer_max_features)
X = vectorizer.fit_transform(df["news"])
y = df["sentiment"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=max_iter)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# Log results to MLflow
with mlflow.start_run():
    mlflow.log_param("model", model_name)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    print("Model and metrics logged successfully in MLflow!")

# Save the model and vectorizer as pickle files
print("Saving the Logistic Regression model and TF-IDF vectorizer...")
with open(model_pkl_path, "wb") as f:
    pickle.dump(model, f)
with open(vectorizer_pkl_path, "wb") as f:
    pickle.dump(vectorizer, f)

# Save sentiment analysis results
df["Logistic_Regression_Sentiment"] = model.predict(X)
print(f"Saving sentiment analysis results to {output_file}...")
df.to_csv(output_file, index=False)

print("Sentiment analysis completed successfully!")
