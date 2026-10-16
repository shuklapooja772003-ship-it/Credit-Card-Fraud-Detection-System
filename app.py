import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata


model, metadata = load_model()
st.title("💳 Credit Card Fraud Detection System")
st.write("A machine learning application for identifying potentially fraudulent credit card transactions.")
st.info(
    "Educational prototype: predictions are risk estimates, not proof of fraud. "
    "Do not use to authorize or decline real payments without appropriate validation."
)

if model is None:
    st.error("Trained model not found. Run `python train_model.py` first.")
    st.stop()

FEATURES = metadata["features"]
THRESHOLD = float(metadata.get("threshold", 0.5))

with st.sidebar:
    st.header("Model information")
    st.write(f"**Algorithm:** {metadata['model_name']}")
    st.write(f"**Decision threshold:** {THRESHOLD:.2f}")
    st.caption("Use the same feature definitions and column order as the training dataset.")


def validate_transactions(data):
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    X = data[FEATURES].copy()
    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="raise")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        raise ValueError("Input contains missing or infinite feature values.")
    if "Amount" in X.columns and (X["Amount"] < 0).any():
        raise ValueError("Transaction Amount cannot be negative.")
    return X


def predict_transactions(data):
    X = validate_transactions(data)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    results = data.copy()
    results["Fraud_Probability"] = probabilities
    results["Prediction"] = np.where(predictions == 1, "Potential Fraud", "Likely Legitimate")
    return results


tab1, tab2, tab3 = st.tabs(["Dashboard", "Single Transaction", "Batch Prediction"])

with tab1:
    st.subheader("System overview")
    st.write("This dashboard summarizes the model configuration. It does not connect to a bank or live payment gateway.")
    col1, col2 = st.columns(2)
    col1.metric("Required model features", len(FEATURES))
    col2.metric("Classification threshold", f"{THRESHOLD:.2%}")
    st.markdown(
        "**How the system works**\n"
        "1. Transaction features are supplied.\n"
        "2. The trained model estimates fraud probability.\n"
        "3. Probability is compared with the decision threshold.\n"
        "4. The result is displayed for review."
    )

with tab2:
    st.subheader("Check an individual transaction")
    st.caption(
        "Enter the numerical features expected by the trained model. "
        "V1–V28 are anonymized PCA features in the standard dataset, not ordinary card details."
    )
    default_row = {feature: 0.0 for feature in FEATURES}
    if "Amount" in FEATURES:
        default_row["Amount"] = 100.0
    input_df = pd.DataFrame([default_row])
    edited_df = st.data_editor(input_df, use_container_width=True, num_rows="fixed", key="single_transaction_editor")
    if st.button("Analyze Transaction", type="primary"):
        try:
            result = predict_transactions(edited_df)
            probability = float(result["Fraud_Probability"].iloc[0])
            prediction = result["Prediction"].iloc[0]
            st.metric("Estimated fraud probability", f"{probability:.2%}")
            if prediction == "Potential Fraud":
                st.error("Potential fraud detected — manual review recommended.")
            else:
                st.success("Classified as likely legitimate at the selected threshold. This does not guarantee safety.")
            st.dataframe(result[["Fraud_Probability", "Prediction"]], use_container_width=True)
        except (ValueError, TypeError) as exc:
            st.error(f"Input error: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab3:
    st.subheader("Batch transaction analysis")
    uploaded_file = st.file_uploader(
        "Upload transaction CSV", type=["csv"],
        help="The file must contain all numerical feature columns used during training."
    )
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data")
            st.dataframe(uploaded_df.head(10), use_container_width=True)
            if st.button("Analyze Uploaded Transactions"):
                results = predict_transactions(uploaded_df)
                total = len(results)
                fraud_count = int((results["Prediction"] == "Potential Fraud").sum())
                legitimate_count = total - fraud_count
                col1, col2, col3 = st.columns(3)
                col1.metric("Transactions analyzed", total)
                col2.metric("Potential fraud", fraud_count)
                col3.metric("Likely legitimate", legitimate_count)
                st.subheader("Prediction results")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download prediction results",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )
        except (ValueError, TypeError) as exc:
            st.error(f"Could not process the uploaded file: {exc}")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

st.divider()
st.caption("Credit Card Fraud Detection | Python + Machine Learning | Academic project")
