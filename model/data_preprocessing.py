import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(
    dataset_path="data/insurance_claims.csv",
    test_size=0.2,
    random_state=42
):
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Drop columns with all missing values
    df = df.dropna(axis=1, how="all")

    # Target column
    target_col = "fraud_reported"

    # Drop irrelevant/ ID-like columns
    drop_cols = [
        "policy_number",
        "policy_bind_date",
        "incident_date",
        "incident_location",
        "insured_zip",
        "auto_make",
        "auto_model",
        "auto_year"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate features and target
    X = df.drop(columns=[target_col])

    # Binary encoding
    y = df[target_col].map({"Y": 1, "N": 0})  

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ]
    )

    # Stratified Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Reset indexes
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, preprocessor
