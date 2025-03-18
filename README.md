# News Ranker & Analyzer

## 📌 Project Overview
This project analyzes financial news articles, classifies them based on sentiment (positive, negative), and ranks them according to their investment impact. The final output provides investors with a ranking system to identify influential news and potential stock movements.

## 📂 Project Structure
```
├── .dvc/                 # DVC tracking files
│   ├── cache/            # Cached DVC files
│   ├── tmp/              # Temporary DVC files
├── .github/workflows/    # CI/CD pipeline configuration
│   ├── mlflow.yml        # Workflow for MLflow tracking
├── config/               # Project configurations
│   ├── config.yaml       # Main configuration file
├── data/                 # Data storage
│   ├── raw/              # Raw financial news data (tracked by DVC)
│   ├── processed/        # Preprocessed data
│   ├── sentiment_results/ # Sentiment analysis results
│   ├── ranking_results/  # Ranked news based on impact
├── deployment/           # Deployment files
│   ├── app.py            # API for model serving
│   ├── Dockerfile        # Containerization setup
│   ├── requirements.txt  # Deployment dependencies
├── mlruns/               # MLflow experiment tracking
├── models/               # Trained models
│   ├── logistic_regression_model.pkl # Logistic regression model
│   ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│   ├── metadata.json     # Model metadata
├── notebooks/            # Jupyter notebooks (currently empty)
├── src/                  # Source code
│   ├── data_collection.py  # Data scraping and collection
│   ├── data_preprocessing.py # Data preprocessing pipeline
│   ├── sentiment_analysis.py # Sentiment analysis using Logistic Regression
│   ├── ranking_system.py  # News ranking algorithm
│   ├── mlflow_tracking.py # MLflow tracking integration
│   ├── main.py           # Main execution pipeline
├── tests/                # Unit tests and validation
│   ├── test_sentiment.py # Test cases for sentiment analysis
│   ├── test_news.csv     # Sample test data
├── .gitignore            # Git ignore rules
├── .dvcignore            # DVC ignore rules
├── dvc.yaml              # DVC pipeline configuration
├── README.md             # Project documentation
```

## 🚀 Key Features
- **Sentiment Analysis using Logistic Regression**: Classifies financial news into positive or negative.
- **News Ranking System**: Assigns a ranking score to financial news based on sentiment impact.
- **Automated Data Processing**: Cleans, tokenizes, and lemmatizes news data for analysis.
- **MLOps Integration**: Supports containerization (Docker), CI/CD workflows, MLflow tracking, and data version control (DVC).

## ⚙️ Installation & Setup
### Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```
### Install dependencies:
```bash
pip install -r requirements.txt
```
### Run the pipeline:
```bash
python src/main.py
```

## 📊 How It Works
### Data Collection (`data_collection.py`)
- Fetches and stores financial news from multiple sources.

### Data Preprocessing (`data_preprocessing.py`)
- Cleans text (removes punctuation, numbers, stopwords).
- Tokenizes and lemmatizes the text.
- Stores processed data in `data/processed/`.

### Sentiment Analysis (`sentiment_analysis.py`)
- Uses **Logistic Regression** with **TF-IDF vectorization**.
- Classifies news sentiment into **positive** or **negative**.
- Saves results in `data/sentiment_results/`.

### News Ranking System (`ranking_system.py`)
- Computes ranking scores using sentiment strength, mention frequency, and market impact.
- Saves ranked news in `data/ranking_results/`.

### Experiment Tracking (`mlflow_tracking.py`)
- Uses **MLflow** to track model performance and experiments.
- Logs hyperparameters, metrics (accuracy, F1-score), and trained models.

## 📌 Data Source
This project uses financial news data from Kaggle:  
[News Sentiment Analysis Dataset](https://www.kaggle.com/datasets/myrios/news-sentiment-analysis/data)

## 📌 Next Steps
- Optimize ranking algorithm using machine learning.
- Integrate real-time stock market data.
- Deploy as a web service for dynamic sentiment tracking.

## 🐜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.
