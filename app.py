import streamlit as st
import pandas as pd
import os
from model.data_preprocessing import load_data
from model.logistic_regression import train_evaluate as lr_model
from model.decision_tree import train_evaluate as dt_model
from model.knn import train_evaluate as knn_model
from model.naive_bayes import train_evaluate as nb_model
from model.random_forest import train_evaluate as rf_model
from model.xgboost import train_evaluate as xgb_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# Page Configuration
st.set_page_config(
    page_title="Auto Insurance Fraud Detection",
    layout="centered"
)

X_train, X_test, y_train, y_test, preprocessor = load_data()

st.title("Auto Insurance Fraud Detection")
st.write(
    """
    This application demonstrates multiple ML 
    classification models to detect fraudulent insurance claims.
    
    \nDataset: Auto Insurance Claims Fraud Dataset (https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data) 
    \nTarget Variable: fraud_reported (Y/N)
    """
)

st.divider()

# Test data download
st.subheader("Download Test Data")

try:
    # Combine test features and target
    test_df = X_test.copy()
    test_df["fraud_reported"] = y_test

    csv = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Test Data",
        data=csv,
        file_name="insurance_test_data.csv",
        mime="text/csv"
    )

except Exception as e:
    st.warning("Test data not available yet.")

st.divider()

# Upload Test Data
st.subheader("Upload Test Data for Evaluation")

uploaded_file = st.file_uploader(
    "Upload the downloaded test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df_uploaded = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Dataset:")
    st.dataframe(df_uploaded.head())

    # Validate dataset
    if "fraud_reported" not in df_uploaded.columns:
        st.error("Uploaded file must contain 'fraud_reported' column.")
        st.stop()

    # Separate features and target
    y_uploaded = df_uploaded["fraud_reported"]

    # Accept BOTH formats: Y/N or 1/0
    if y_uploaded.dtype == "object":
        y_uploaded = (
            y_uploaded.astype(str)
            .str.strip()
            .str.upper()
            .map({"Y": 1, "N": 0})
        )

    # Convert safely to numeric
    y_uploaded = pd.to_numeric(y_uploaded, errors="coerce")

    # Stop if invalid values exist
    if y_uploaded.isna().any():
        st.error(
            "Invalid values found in 'fraud_reported'. "
            "Expected Y/N or 1/0."
        )
        st.stop()

    X_uploaded = df_uploaded.drop(columns=["fraud_reported"])

    st.success("Dataset successfully prepared for evaluation.")

st.divider()

# Model Selection
st.subheader("Select Classification Model")

model_name = st.selectbox(
    "Choose a model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_descriptions = {
    "Logistic Regression": "Linear classifier suitable for binary classification problems.",
    "Decision Tree": "Tree-based model that splits data using decision rules.",
    "KNN": "Instance-based learning algorithm using nearest neighbors.",
    "Naive Bayes": "Probabilistic classifier based on Bayes theorem.",
    "Random Forest": "Ensemble of decision trees improving stability and performance.",
    "XGBoost": "Optimized gradient boosting algorithm for high predictive power."
}

st.info(model_descriptions[model_name])

# Train & Evaluate
if st.button("Train & Evaluate Model"):

    if uploaded_file is None:
        st.warning("Please upload the downloaded test dataset first.")
        st.stop()

    with st.spinner("Training model....."):
        if model_name == "Logistic Regression":
            results = lr_model()
            model = results["model"]

        elif model_name == "Decision Tree":
            results = dt_model()
            model = results["model"]

        elif model_name == "KNN":
            results = knn_model()
            model = results["model"]

        elif model_name == "Naive Bayes":
            results = nb_model()
            model = results["model"]

        elif model_name == "Random Forest":
            results = rf_model()
            model = results["model"]

        elif model_name == "XGBoost":
            results = xgb_model()
            model = results["model"]

    st.success("Model evaluation completed successfully!")

    st.subheader("Model Performance Metrics")

    # Evaluate on uploaded dataset
    y_pred = model.predict(X_uploaded)

    # Probabilities for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_uploaded)[:, 1]
    else:
        y_prob = y_pred

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_uploaded, y_pred),
        "AUC Score": roc_auc_score(y_uploaded, y_prob),
        "Precision": precision_score(y_uploaded, y_pred),
        "Recall": recall_score(y_uploaded, y_pred),
        "F1 Score": f1_score(y_uploaded, y_pred),
        "MCC": matthews_corrcoef(y_uploaded, y_pred)
    }

    # Show metrics
    results_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    
    st.table(results_df)

    st.subheader("Model Interpretation")

    accuracy = metrics["Accuracy"]
    precision = metrics["Precision"]
    recall = metrics["Recall"]
    f1 = metrics["F1 Score"]
    auc = metrics["AUC Score"]

    # Overall Performance
    if auc >= 0.85:
        st.success("High AUC: Excellent model discrimination capability.")
    elif auc >= 0.75:
        st.info("Good AUC: Good discrimination between fraud and non-fraud cases.")
    else:
        st.warning("Low AUC: Limited ability to separate fraud from non-fraud cases.")

    # Fraud Detection Strength
    if recall >= 0.75:
        st.success("High recall: Most fraudulent claims are correctly detected.")
    elif recall >= 0.60:
        st.info("Moderate recall: Some fraudulent cases may still be missed.")
    else:
        st.warning("Low recall: Many fraudulent cases may go undetected.")

    # False Alarm Risk
    if precision >= 0.70:
        st.success("High Precision (Low false alarm rate): Most flagged claims are truly fraudulent.")
    elif precision >= 0.50:
        st.info("Moderate precision: Some legitimate claims may be incorrectly flagged.")
    else:
        st.warning("Low Precision: (High false alarm rate): Many legitimate claims may be flagged as fraud.")

    # Balance Between Precision & Recall
    if f1 >= 0.70:
        st.success("High F1 Score: Good balance between detecting fraud and minimizing false alarms.")
    else:
        st.info("Low F1 Score: The model may favor either recall or precision. Threshold tuning could improve balance.")

    st.write(
        "In fraud detection, recall is often prioritized because failing to detect fraud can result in financial losses."
    )
    if recall > precision:
        st.write("This model prioritizes catching fraud over avoiding false alarms.")
    elif precision > recall:
        st.write("This model is conservative, reducing false alarms but potentially missing fraud.")
    else:
        st.write("This model maintains a balanced fraud detection strategy.")

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_uploaded, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
