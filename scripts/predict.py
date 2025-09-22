
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.5

def main():
    parser = argparse.ArgumentParser(description="Score patients for 30-day readmission risk.")
    parser.add_argument("--model_path", type=str, default="models/random_forest.joblib")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with the same schema as training features.")
    parser.add_argument("--output_csv", type=str, default="reports/predictions.csv")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first with src/train.py.")

    clf = joblib.load(model_path)

    df = pd.read_csv(args.input_csv).replace("?", np.nan)
    for col in ["encounter_id", "patient_nbr", "readmitted"]:
        if col in df.columns:
            df = df.drop(columns=col)

    proba = clf.predict_proba(df)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    out = df.copy()
    out["readmit_proba"] = proba
    out["readmit_pred"] = pred

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

if __name__ == "__main__":
    main()
