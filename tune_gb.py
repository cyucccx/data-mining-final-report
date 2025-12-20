import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from itertools import product
import time

print("===== Loading Processed Data =====")

# ============================================================
# 1. Load processed data
# ============================================================
X_train = pd.read_csv("processed-data/X_train_processed.csv")
X_valid = pd.read_csv("processed-data/X_valid_processed.csv")
y_train = pd.read_csv("processed-data/y_train.csv").values.ravel()
y_valid = pd.read_csv("processed-data/y_valid.csv").values.ravel()

print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

# ============================================================
# 2. Define parameter grid
# ============================================================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 1.0],
}

# ============================================================
# 3. Quick search with key parameters first
# ============================================================
print("\n===== Phase 1: Quick Parameter Search =====")
print("Testing key parameters: n_estimators, max_depth, learning_rate\n")

quick_results = []

# Test key parameters
for n_est in [100, 200, 300]:
    for depth in [3, 5, 7]:
        for lr in [0.05, 0.1, 0.2]:
            start = time.time()
            
            model = GradientBoostingClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_valid)
            y_proba = model.predict_proba(X_valid)[:, 1]
            
            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_proba)
            
            elapsed = time.time() - start
            
            result = {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
                "time": elapsed
            }
            quick_results.append(result)
            
            print(f"n_est={n_est:3d}, depth={depth}, lr={lr:.2f} | "
                  f"F1: {f1:.4f}, AUC: {auc:.4f}, Acc: {acc:.4f} ({elapsed:.1f}s)")

# Find best from phase 1
quick_df = pd.DataFrame(quick_results)
best_quick = quick_df.loc[quick_df["f1"].idxmax()]
print(f"\n✓ Best Phase 1: n_est={int(best_quick['n_estimators'])}, "
      f"depth={int(best_quick['max_depth'])}, lr={best_quick['learning_rate']:.2f}")
print(f"  F1: {best_quick['f1']:.4f}, AUC: {best_quick['roc_auc']:.4f}")

# ============================================================
# 4. Fine-tune with additional parameters
# ============================================================
print("\n===== Phase 2: Fine-tuning =====")
print("Testing min_samples_split, min_samples_leaf, subsample\n")

best_n_est = int(best_quick["n_estimators"])
best_depth = int(best_quick["max_depth"])
best_lr = best_quick["learning_rate"]

fine_results = []

for min_split in [2, 5, 10]:
    for min_leaf in [1, 2, 4]:
        for subsample in [0.8, 0.9, 1.0]:
            start = time.time()
            
            model = GradientBoostingClassifier(
                n_estimators=best_n_est,
                max_depth=best_depth,
                learning_rate=best_lr,
                min_samples_split=min_split,
                min_samples_leaf=min_leaf,
                subsample=subsample,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_valid)
            y_proba = model.predict_proba(X_valid)[:, 1]
            
            acc = accuracy_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            auc = roc_auc_score(y_valid, y_proba)
            
            elapsed = time.time() - start
            
            result = {
                "n_estimators": best_n_est,
                "max_depth": best_depth,
                "learning_rate": best_lr,
                "min_samples_split": min_split,
                "min_samples_leaf": min_leaf,
                "subsample": subsample,
                "accuracy": acc,
                "f1": f1,
                "roc_auc": auc,
                "time": elapsed
            }
            fine_results.append(result)
            
            print(f"split={min_split:2d}, leaf={min_leaf}, sub={subsample:.1f} | "
                  f"F1: {f1:.4f}, AUC: {auc:.4f}, Acc: {acc:.4f}")

# ============================================================
# 5. Summary
# ============================================================
print("\n" + "=" * 60)
print("===== RESULTS SUMMARY =====")
print("=" * 60)

# Combine all results
all_results = quick_results + fine_results
results_df = pd.DataFrame(all_results)

# Top 10 by F1
print("\nTop 10 Configurations by F1 Score:")
print("-" * 60)
top10 = results_df.nlargest(10, "f1")
for i, row in top10.iterrows():
    params = f"n={int(row['n_estimators'])}, d={int(row['max_depth'])}, lr={row['learning_rate']:.2f}"
    if "min_samples_split" in row and pd.notna(row.get("min_samples_split")):
        params += f", split={int(row['min_samples_split'])}, leaf={int(row['min_samples_leaf'])}, sub={row['subsample']:.1f}"
    print(f"F1: {row['f1']:.4f} | AUC: {row['roc_auc']:.4f} | {params}")

# Best overall
best = results_df.loc[results_df["f1"].idxmax()]
print("\n" + "=" * 60)
print("✓ BEST CONFIGURATION:")
print("=" * 60)
print(f"  n_estimators:     {int(best['n_estimators'])}")
print(f"  max_depth:        {int(best['max_depth'])}")
print(f"  learning_rate:    {best['learning_rate']}")
if "min_samples_split" in best and pd.notna(best.get("min_samples_split")):
    print(f"  min_samples_split: {int(best['min_samples_split'])}")
    print(f"  min_samples_leaf:  {int(best['min_samples_leaf'])}")
    print(f"  subsample:         {best['subsample']}")
print(f"\n  Accuracy:  {best['accuracy']:.4f}")
print(f"  F1 Score:  {best['f1']:.4f}")
print(f"  ROC-AUC:   {best['roc_auc']:.4f}")

# Save results
results_df.to_csv("gb_tuning_results.csv", index=False)
print("\nResults saved to gb_tuning_results.csv")

print("\n===== DONE =====")
