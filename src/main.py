from src.data_collection import collect_data  
from src.data_processing import load_and_process_data
from src.sentiment_analysis import analyze_sentiment
from src.ranking_system import rank_stocks

def main():
    print("Starting End-to-End ML Pipeline...")

    # Step 1: Data Collection
    print("Collecting Data...")
    collect_data()
    
    # Step 2: Data Processing
    print("Processing Data...")
    load_and_process_data()
    
    # Step 3: Sentiment Analysis using Logistic Regression
    print("Running Logistic Regression Sentiment Analysis...")
    analyze_sentiment()
    
    # Step 4: Ranking Stocks based on Sentiment Analysis
    print("Ranking Stocks...")
    rank_stocks()
    
    print("Pipeline Execution Completed!")

if __name__ == "__main__":
    main()