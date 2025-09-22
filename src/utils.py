
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TARGET_COL = "readmitted"
ID_COLS = ["encounter_id", "patient_nbr"]

def load_and_preprocess_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    for c in ID_COLS:
        if c in df.columns:
            df = df.drop(columns=c)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found.")
    y = df[TARGET_COL].apply(lambda x: 1 if x == "<30" else 0).astype(int)
    X = df.drop(columns=[TARGET_COL])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)]
    )

    return X, y, preprocessor, num_cols + cat_cols

def split_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def compute_metrics(y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = float("nan")
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "roc_auc": float(roc), "confusion_matrix": cm}

def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
