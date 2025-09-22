
from pathlib import Path
import joblib

def main():
    model_path = Path("models/random_forest.joblib")
    if not model_path.exists():
        raise FileNotFoundError("Trained model not found at models/random_forest.joblib. Run src/train.py first.")
    _ = joblib.load(model_path)
    print("Model loaded. Extend this script to load persisted test data if desired.")

if __name__ == "__main__":
    main()
