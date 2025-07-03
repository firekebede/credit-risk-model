
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from mlflow.models.signature import infer_signature

# === Load the processed labeled data ===
df = pd.read_csv("data/processed/df_labeled.csv")

# === Feature selection and preprocessing ===
columns_to_drop = ["CustomerId", "TransactionId", "BatchId", "SubscriptionId", "AccountId"]
X = df.drop(columns=["is_high_risk"] + columns_to_drop, errors='ignore')
X = X.select_dtypes(include=["number"]).fillna(0)
y = df["is_high_risk"]

# === Standardize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Set MLflow experiment ===
mlflow.set_experiment("credit-risk-model")

def train_and_log_model(model, model_name, params):
    with mlflow.start_run(run_name=model_name):
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_param("model", model_name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc
        })

        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Log model and register in MLflow
        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="credit_risk_classifier"
        )

        print(f"\n--- {model_name} Report ---")
        print(classification_report(y_test, preds))

# === Train Logistic Regression ===
lr = LogisticRegression(solver='liblinear', random_state=42)
train_and_log_model(lr, "LogisticRegression", {"C": 1.0})

# === Train Random Forest ===
rf = RandomForestClassifier(random_state=42)
train_and_log_model(rf, "RandomForest", {"n_estimators": 100, "max_depth": 5})

print("\n✅ Training and model registration complete.")
