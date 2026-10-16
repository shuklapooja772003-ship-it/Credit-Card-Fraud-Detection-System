# Credit Card Fraud Detection System

A Python and machine-learning academic project for classifying credit card transactions as potentially fraudulent or likely legitimate.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## Dataset
Download the public Credit Card Fraud Detection dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the extracted `creditcard.csv` file in the `data/` directory. The dataset is not included in this ZIP.

## Installation (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train models
```bash
python train_model.py
```

This creates the trained model and evaluation files in `models/` and `outputs/`.

## Run the web application
```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually http://localhost:8501.

## Target variable
- `Class = 0`: legitimate transaction
- `Class = 1`: fraudulent transaction

## Evaluation
The project compares Logistic Regression and Random Forest using precision, recall, F1-score, ROC-AUC, and PR-AUC. The model with the higher held-out test PR-AUC is selected for this educational demonstration.

## Limitations
Predictions are educational risk estimates, not proof of fraud. The default probability threshold is illustrative and should be tuned using validation data. Real deployment requires representative data, robust validation, monitoring, security controls, and human review.
