import os
from dotenv import load_dotenv
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from mlflow.models.signature import infer_signature
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

def evaluate_model(model, X, y, prefix="") -> dict:
    """
    Evaluates a classification model and returns metrics.

    Parameters:
    model: Trained classification model.
    X (pd.DataFrame): Features to predict on.
    y (pd.Series): True labels.
    prefix (str): Optional prefix for metric keys (e.g., 'train_', 'test_').

    Returns:
    dict: Dictionary of evaluation metrics.

    Raises:
    ValueError: If prediction fails due to mismatched input.
    """
    try:
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)

        metrics = {
            f"{prefix}score"            : model.score(X, y),
            f"{prefix}accuracy_score"   : accuracy_score(y, y_pred),
            f"{prefix}balanced_accuracy": balanced_accuracy_score(y, y_pred),
            f"{prefix}log_loss"         : log_loss(y, y_proba),
            f"{prefix}precision_score"  : precision_score(y, y_pred, average='weighted'),
            f"{prefix}recall_score"     : recall_score(y, y_pred, average='weighted'),
            f"{prefix}f1_score"         : f1_score(y, y_pred, average='weighted'),
            f"{prefix}matthews_corrcoef": matthews_corrcoef(y, y_pred)
        }

        if len(set(y)) > 2:
            y_bin = label_binarize(y, classes=sorted(set(y)))
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
        else:
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y, y_proba[:, 1])

        return metrics

    except Exception as e:
        raise ValueError(f"[ERROR] Model evaluation failed: {str(e)}") from e


def main():
    """
    Trains and tunes a RandomForest model using GridSearchCV,
    and manually logs training/testing metrics and model using MLflow.
    """
    try:
        # Autentikasi ke DagsHub
        load_dotenv()
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not username or not password:
            raise EnvironmentError("MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD harus di-set sebagai environment variable")

        # Set MLflow tracking to DagsHub (kriteria advanced)
        mlflow.set_tracking_uri("https://dagshub.com/mellisadmyn/sleep_disorder_mlflow.mlflow")
        mlflow.set_experiment("Sleep_Disorder_Classification_RF")

        # Set MLflow tracking to local (kriteria skilled)
        # mlflow.set_tracking_uri("file:./mlruns")
        # mlflow.set_experiment("Sleep_Disorder_RF")

        X_train = load_data("processed-dataset/X_train.csv")
        X_test = load_data("processed-dataset/X_test.csv")
        y_train = load_data("processed-dataset/y_train.csv").squeeze()
        y_test = load_data("processed-dataset/y_test.csv").squeeze()

        with mlflow.start_run(run_name="rf_manuallog_hyptuning-params"):
            param_grid = {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 3, 5],
                "min_samples_leaf": [1, 2, 3]
            }

            base_model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, n_jobs=-1, scoring="accuracy", verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            print(f"[INFO] Best params: {grid_search.best_params_}")
            mlflow.log_params(grid_search.best_params_)

            # Manual log training metrics
            train_metrics = evaluate_model(best_model, X_train, y_train, prefix="train_")
            for key, value in train_metrics.items():
                mlflow.log_metric(key, value)
            print("[INFO] Training metrics logged manually.")

            # Manual log testing metrics
            test_metrics = evaluate_model(best_model, X_test, y_test, prefix="test_")
            for key, value in test_metrics.items():
                mlflow.log_metric(key, value)
            print("[INFO] Testing metrics logged manually.")

            # Log model signature
            input_example = X_train.iloc[:1]
            signature = infer_signature(X_train, best_model.predict(X_train))

            # Log model artifact
            mlflow.sklearn.log_model(
                best_model,
                artifact_path="random_forest_model",
                input_example=input_example,
                signature=signature
            )
            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("model", "RandomForestClassifier")
            mlflow.set_tag("evaluation_extended", "True")
            mlflow.set_tag("extra_metrics", "MCC, Balanced Accuracy") # min 2 metrik tambahan di luar autolog
            print("[INFO] Best model logged.")

            run_id = mlflow.active_run().info.run_id
            print(f"[INFO] Run ID: {run_id}")
            print("[INFO] Hyperparameter tuning and logging completed.")

    except Exception as e:
        print(f"[ERROR] Modelling execution failed: {str(e)}")

if __name__ == "__main__":
    main()
