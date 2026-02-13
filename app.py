import streamlit as st
import pandas as pd
import os
from model.logistic_regression import train_evaluate as lr_model
from model.decision_tree import train_evaluate as dt_model
from model.knn import train_evaluate as knn_model
from model.naive_bayes import train_evaluate as nb_model
from model.random_forest import train_evaluate as rf_model
from model.xgboost import train_evaluate as xgb_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Auto Insurance Fraud Detection",
    layout="centered"
)

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

# Dataset Upload
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file to preview",
    type=["csv"]
)

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Dataset:")
    st.dataframe(df_uploaded.head())

    #Class Distribution
    st.subheader("Class Distribution")

    if "fraud_reported" in df_uploaded.columns:
        
        fraud_counts = df_uploaded["fraud_reported"].value_counts()

        fig2, ax2 = plt.subplots()
        fraud_counts.plot(kind="bar", ax=ax2)
        ax2.set_title("Fraud vs Non-Fraud Distribution")
        ax2.set_ylabel("Count")

        st.pyplot(fig2)

        st.write(
            "Fraud datasets are typically imbalanced. "
            "This affects model evaluation and requires careful metric selection."
        )

    else:
        st.info(
            "The uploaded dataset does not contain a 'fraud_reported' column. "
            "Class distribution analysis is skipped."
        )

st.divider()

# Dataset Download
st.subheader("Download Dataset")

DATA_PATH = os.path.join("data", "insurance_claims.csv")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        st.download_button(
            label="Download Auto Insurance Dataset",
            data=f,
            file_name="insurance_claims.csv",
            mime="text/csv"
        )
else:
    st.warning("Dataset file not found in the data folder.")

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

    with st.spinner("Training model....."):
        if model_name == "Logistic Regression":
            results = lr_model()

        elif model_name == "Decision Tree":
            results = dt_model()

        elif model_name == "KNN":
            results = knn_model()

        elif model_name == "Naive Bayes":
            results = nb_model()

        elif model_name == "Random Forest":
            results = rf_model()

        elif model_name == "XGBoost":
            results = xgb_model()

    st.success("Model evaluation completed successfully!")

    st.subheader("Model Performance Metrics")

    # Extract returned values
    metrics = results["metrics"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]

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

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)
