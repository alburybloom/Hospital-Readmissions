
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from utils import (RANDOM_STATE, compute_metrics, load_and_preprocess_csv,
                   save_json, split_data)

def main(args):
    X, y, preprocessor, _ = load_and_preprocess_csv(args.data_path)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size, random_state=RANDOM_STATE)
    clf = Pipeline(steps=[("prep", preprocessor),
                         ("rf", RandomForestClassifier(n_estimators=args.n_estimators,
                                                       max_depth=args.max_depth,
                                                       random_state=RANDOM_STATE,
                                                       class_weight="balanced",
                                                       n_jobs=-1))])
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= args.threshold).astype(int)

    metrics = compute_metrics(y_test, y_proba, y_pred)
    print("Metrics:", metrics)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, "models/random_forest.joblib")
    save_json(metrics, "reports/metrics.json")

    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix (threshold = {args.threshold:.2f})")
    fig.tight_layout()
    fig.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close(fig)

    try:
        result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring="f1")
        import pandas as pd
        imp = pd.DataFrame({"feature_index": range(len(result.importances_mean)),
                            "importance_mean": result.importances_mean,
                            "importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
        imp.to_csv("reports/feature_importance.csv", index=False)
    except Exception as e:
        print("Permutation importance failed:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest for 30-day readmissions")
    parser.add_argument("--data_path", type=str, default="data/raw/diabetes_130_us_hospitals.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
