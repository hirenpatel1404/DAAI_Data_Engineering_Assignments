"""
Assignment 9: ML pipeline orchestrated with Apache Airflow
Student: Hiren Patel

Pipeline (one task per step):
load_data -> clean_data -> split_data -> save_data -> train_model -> evaluate_model

Tasks run as separate processes, so I pass data between them with files in a shared
folder (DATA_DIR) and, for the split, with Airflow XCom (cross-communication).
"""
import os
import json
import pickle
from datetime import datetime, timedelta

import pandas as pd

# I support both Airflow 3 (new import paths) and Airflow 2
try:
    from airflow.sdk import DAG
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

# I keep all pipeline files in a folder that is mounted from my laptop
DATA_DIR = os.environ.get("IRIS_DATA_DIR", "/opt/airflow/data")
RAW_PATH = os.path.join(DATA_DIR, "iris_raw.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "iris_clean.csv")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")


def load_data():
    # I load the Iris dataset from scikit-learn and save it as the raw CSV
    from sklearn.datasets import load_iris
    os.makedirs(DATA_DIR, exist_ok=True)
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df.to_csv(RAW_PATH, index=False)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns -> {RAW_PATH}")
    return df.shape[0]


def clean_data():
    # I remove duplicate rows and missing values, and check that measurements are positive
    df = pd.read_csv(RAW_PATH)
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    feature_cols = [c for c in df.columns if c != "species"]
    df = df[(df[feature_cols] > 0).all(axis=1)]
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaning: {before} rows -> {len(df)} rows ({before - len(df)} removed) -> {CLEAN_PATH}")
    return len(df)


def split_data(**context):
    # I split into 80% train and 20% test, keeping the class balance (stratify)
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(CLEAN_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=0, stratify=df["species"])
    print(f"Split: {len(train_df)} train rows, {len(test_df)} test rows")
    # I push the two parts to XCom as lists of rows so the next task can pick them up
    context["ti"].xcom_push(key="train", value=train_df.to_dict(orient="records"))
    context["ti"].xcom_push(key="test", value=test_df.to_dict(orient="records"))


def save_data(**context):
    # I pull the split from XCom and save it as train.csv and test.csv
    ti = context["ti"]
    train_df = pd.DataFrame(ti.xcom_pull(task_ids="split_data", key="train"))
    test_df = pd.DataFrame(ti.xcom_pull(task_ids="split_data", key="test"))
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved {TRAIN_PATH} ({len(train_df)} rows) and {TEST_PATH} ({len(test_df)} rows)")


def train_model():
    # I train a Logistic Regression model inside a scaling pipeline and save it with pickle
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train_df = pd.read_csv(TRAIN_PATH)
    X, y = train_df.drop(columns="species"), train_df["species"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained on {len(X)} rows, training accuracy {model.score(X, y):.3f} -> {MODEL_PATH}")


def evaluate_model():
    # I load the saved model, test it on unseen data and save the metrics as JSON
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    test_df = pd.read_csv(TEST_PATH)
    X, y = test_df.drop(columns="species"), test_df["species"]
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    report = classification_report(y, pred, target_names=["setosa", "versicolor", "virginica"], output_dict=True)
    metrics = {
        "test_rows": int(len(y)),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(classification_report(y, pred, target_names=["setosa", "versicolor", "virginica"]))
    print(f"Test accuracy: {acc:.3f} -> {METRICS_PATH}")
    if acc < 0.8:
        # I fail the task on purpose if the model is too weak, so Airflow marks the run red
        raise ValueError(f"Accuracy {acc:.3f} is below the 0.80 quality gate")
    return metrics["accuracy"]


default_args = {
    "owner": "hiren",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="iris_ml_pipeline",
    description="Load, clean, split, save, train and evaluate an Iris classifier",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,          # I trigger it manually from the UI
    catchup=False,
    tags=["assignment9", "ml"],
) as dag:
    t1 = PythonOperator(task_id="load_data", python_callable=load_data)
    t2 = PythonOperator(task_id="clean_data", python_callable=clean_data)
    t3 = PythonOperator(task_id="split_data", python_callable=split_data)
    t4 = PythonOperator(task_id="save_data", python_callable=save_data)
    t5 = PythonOperator(task_id="train_model", python_callable=train_model)
    t6 = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    # I set the order of the tasks
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
