import mlflow
import mlflow.sklearn

def start_mlflow_experiment(experiment_name):
    """Initialize MLflow experiment."""
    mlflow.set_experiment(experiment_name)

def log_model_params(model_name, max_iter):
    """Log hyperparameters in MLflow."""
    mlflow.log_param("model", model_name)
    mlflow.log_param("max_iter", max_iter)

def log_metrics(accuracy, f1_score):
    """Log model performance metrics in MLflow."""
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1_score)

def save_model(model, model_name):
    """Save the trained model in MLflow."""
    mlflow.sklearn.log_model(model, model_name)
