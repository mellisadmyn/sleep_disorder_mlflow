import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

def load_data(path: str):
    return pd.read_csv(path)

def main():
    # Set MLflow tracking ke lokal
    mlflow.set_tracking_uri("file:./mlruns") 
    mlflow.set_experiment("Sleep_Disorder_RF-autolog")

    # Load data
    df  = load_data("clean_sleep_health_and_lifestyle_dataset.csv")
    X   = df.drop(columns=["sleep_disorder"])
    y   = df["sleep_disorder"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Autolog
    mlflow.sklearn.autolog()

    # Model training & logging
    with mlflow.start_run(run_name="random_forest_autolog"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Model trained and autologged.")

if __name__ == "__main__":
    main()
