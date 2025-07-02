
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler

# === Load the labeled data ===
df = pd.read_csv("data/processed/df_labeled.csv")

# === Features and target ===
X = df.drop(columns=["is_high_risk", "CustomerId"], errors='ignore')
y = df["is_high_risk"]
X = X.fillna(0)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split into training and testing sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# === Start MLflow experiment ===
mlflow.set_experiment("credit-risk-model")

def train_and_log_model(model, model_name, params):
    with mlflow.start_run(run_name=model_name):
        if params:
            model.set_params(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # === Metrics ===
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

        mlflow.sklearn.log_model(model, "model")

        print(f"\n--- {model_name} Report ---")
        print(classification_report(y_test, preds))

# === Train Logistic Regression ===
lr = LogisticRegression(solver='liblinear', random_state=42)
lr_params = {"C": 1.0}
train_and_log_model(lr, "LogisticRegression", lr_params)

# === Train Random Forest ===
rf = RandomForestClassifier(random_state=42)
rf_params = {"n_estimators": 100, "max_depth": 5}
train_and_log_model(rf, "RandomForest", rf_params)

print("\n✅ Training complete. Visit http://localhost:5000 to view MLflow logs.")
