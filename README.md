# Predicting 30-Day Hospital Readmissions (UCI Diabetes)

[![CI](https://img.shields.io/github/actions/workflow/status/alburybloom/readmissions_rf/ci.yml?branch=main)](https://github.com/alburybloom/readmissions_rf/actions)

This repository provides an end-to-end pipeline to predict 30-day hospital readmissions using the UCI Diabetes readmission dataset. It includes robust preprocessing, a Random Forest baseline, reproducible training, a quickstart notebook, metrics and figures, CI, and a CLI for batch scoring.

## Project Structure
```
readmissions_rf/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── predict_readmissions.ipynb
├── reports/
│   └── figures/
├── scripts/
│   └── predict.py
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   └── test_preprocess.py
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Place dataset at: data/raw/diabetes_130_us_hospitals.csv
python src/train.py --data_path data/raw/diabetes_130_us_hospitals.csv --n_estimators 300 --max_depth 20 --threshold 0.50
python scripts/predict.py --input_csv data/raw/diabetes_130_us_hospitals.csv --output_csv reports/predictions.csv --threshold 0.50
```
