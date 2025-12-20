import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score

print("===== Loading Processed Data =====")

# ============================================================
# 1. Load processed data
# ============================================================
X_train = pd.read_csv("processed-data/X_train_processed.csv")
X_valid = pd.read_csv("processed-data/X_valid_processed.csv")
X_test = pd.read_csv("processed-data/X_test_processed.csv")
y_train = pd.read_csv("processed-data/y_train.csv").values.ravel()
y_valid = pd.read_csv("processed-data/y_valid.csv").values.ravel()
test_id = pd.read_csv("processed-data/test_id.csv").values.ravel()

print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}, Test shape: {X_test.shape}")

# ============================================================
# 2. Define models to evaluate
# ============================================================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
}

# ============================================================
# 3. Train and evaluate each model
# ============================================================
print("\n===== Model Evaluation =====")
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Validation predictions
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)
    auc = roc_auc_score(y_valid, y_proba) if y_proba is not None else None
    
    results[name] = {"accuracy": acc, "f1": f1, "auc": auc, "model": model}
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    if auc:
        print(f"  ROC-AUC:  {auc:.4f}")

# ============================================================
# 4. Select best model based on F1 score
# ============================================================
best_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_name]["model"]
print(f"\n===== Best Model: {best_name} =====")
print(f"  F1: {results[best_name]['f1']:.4f}")
print(f"  Accuracy: {results[best_name]['accuracy']:.4f}")

# ============================================================
# 5. Retrain best model on full train + valid data
# ============================================================
print("\n===== Retraining on Full Data =====")
X_full = pd.concat([X_train, X_valid], axis=0)
y_full = np.concatenate([y_train, y_valid])

# Clone and retrain
if best_name == "LogisticRegression":
    final_model = LogisticRegression(max_iter=1000, random_state=42)
elif best_name == "RandomForest":
    final_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
else:
    final_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

final_model.fit(X_full, y_full)
print("Final model trained on full dataset.")

# ============================================================
# 6. Generate predictions for test set
# ============================================================
print("\n===== Generating Predictions =====")
test_predictions = final_model.predict(X_test)

# ============================================================
# 7. Create submission file
# ============================================================
submission = pd.DataFrame({
    "id": test_id,
    "Depression": test_predictions
})

submission.to_csv("submission.csv", index=False)
print(f"\nSubmission saved to submission.csv")
print(f"Prediction distribution:\n{pd.Series(test_predictions).value_counts()}")

# ============================================================
# 8. Feature importance (if available)
# ============================================================
if hasattr(final_model, "feature_importances_"):
    print("\n===== Top 10 Feature Importances =====")
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(importance_df.head(10).to_string(index=False))

print("\n===== DONE =====")