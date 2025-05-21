import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a CSV file as a DataFrame.

    Parameters:
    path (str): File path to the CSV.

    Returns:
    pd.DataFrame: Loaded dataset.

    Raises:
    FileNotFoundError: If the file does not exist.
    pd.errors.ParserError: If the file cannot be parsed.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[ERROR] File not found: {path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"[ERROR] Could not parse the file: {path}") from e

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates a classification model on the test dataset.

    Parameters:
    model: Trained classification model.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): True labels for test set.

    Returns:
    dict: Dictionary of evaluation metrics.

    Raises:
    ValueError: If prediction fails due to mismatched input.
    """
    try:
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = {
            "score"     : model.score(X_test, y_test),
            "accuracy"  : accuracy_score(y_test, y_pred),
            "log_loss"  : log_loss(y_test, y_proba),
            "precision" : precision_score(y_test, y_pred, average='weighted'),
            "recall"    : recall_score(y_test, y_pred, average='weighted'),
            "f1_score"  : f1_score(y_test, y_pred, average='weighted'),
        }

        # ROC AUC
        if len(set(y_test)) > 2:
            y_bin = label_binarize(y_test, classes=sorted(set(y_test)))
            metrics["roc_auc"] = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])

        return metrics
    
    except Exception as e:
        raise ValueError(f"[ERROR] Model evaluation failed: {str(e)}") from e

def main():
    """
    Loads preprocessed dataset, trains a RandomForestClassifier,
    logs the experiment using MLflow autologging, and evaluates the model.

    Raises:
    Exception: For any failure during the modelling execution.
    """
    try:
        # Set MLflow tracking to local
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Sleep_Disorder_RF")

        # Load datasets
        X_train = load_data("processed-dataset/X_train.csv")
        X_test  = load_data("processed-dataset/X_test.csv")
        y_train = load_data("processed-dataset/y_train.csv").squeeze()
        y_test  = load_data("processed-dataset/y_test.csv").squeeze()

        # Enable autologging
        mlflow.sklearn.autolog()

        # Train model with autolog
        with mlflow.start_run(run_name="rf_autolog_fixed-params"):
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Autolog logs training metrics automatically
            run_id = mlflow.active_run().info.run_id
            print(f"[INFO] Run ID: {run_id}")
            print("[INFO] Training metrics have been autologged by MLflow.")

            # Manual logging for test metrics
            metrics = evaluate_model(model, X_test, y_test)
            for key, value in metrics.items():
                mlflow.log_metric(f"testing_{key}", value)
            print("[INFO] Testing metrics have been autologged by MLflow.")

            print("[INFO] Model trained and autologged with MLflow.")

    except Exception as e:
        print(f"[ERROR] Modelling execution failed: {str(e)}")

if __name__ == "__main__":
    main()
