
import pandas as pd
from src.utils import load_and_preprocess_csv, TARGET_COL

def test_target_mapping(tmp_path):
    df = pd.DataFrame({
        'readmitted': ['<30', '>30', 'NO', '<30', 'NO'],
        'age': [60, 70, 80, 50, 40],
        'gender': ['Male', 'Female', 'Female', 'Male', 'Unknown/Invalid']
    })
    p = tmp_path / "toy.csv"
    df.to_csv(p, index=False)

    X, y, preprocessor, feats = load_and_preprocess_csv(str(p))
    assert set(y.unique()) == {0, 1}
    assert y.sum() == 2
    assert TARGET_COL not in X.columns
