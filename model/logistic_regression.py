from model.data_preprocessing import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

def train_evaluate():

    # Load data
    X_train, X_test, y_train, y_test, preprocessor = load_data()

    # Create downloadable test dataset
    test_df = X_test.copy()
    test_df["fraud_reported"] = y_test

    # Build pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=0.5,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Predict probabilities (for AUC)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred
    }
