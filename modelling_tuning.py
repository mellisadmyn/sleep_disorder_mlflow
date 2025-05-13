import os
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, log_loss, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

def load_data(path: str):
    return pd.read_csv(path)

def train_model(X_train, y_train, params):
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

def main():
    # Set MLflow tracking lokal
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Sleep_Disorder_Classification_XGBoost")

    # Load data
    df  = load_data("clean_sleep_health_and_lifestyle_dataset.csv")
    X   = df.drop(columns=["sleep_disorder"])
    y   = df["sleep_disorder"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning grid
    param_grid = [
        {"n_estimators": n, "max_depth": d, "learning_rate": lr, "random_state": 42}
        for n in [100, 200]
        for d in [3, 6]
        for lr in [0.01, 0.1]
    ]

    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f"xgb_tuning_run_{i+1}"):
            model = train_model(X_train, y_train, params)
            metrics = evaluate_model(model, X_test, y_test)

            # Manual logging
            mlflow.log_params(params)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            mlflow.set_tag("stage", "tuning")
            mlflow.set_tag("model", "XGBClassifier")

            # Save model
            mlflow.sklearn.log_model(model, artifact_path="xgb_model")

            print(f"[Run {i+1}] Model trained and logged with params: {params}")

if __name__ == "__main__":
    main()
