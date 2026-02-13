from model.data_preprocessing import load_data
from sklearn.naive_bayes import GaussianNB
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

    # Create pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GaussianNB())
    ])

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred
    }
