import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# 1. 載入經過 data-preprocessing.py 處理過的資料
# 確保這些檔案在你的目錄中
X_train = pd.read_csv('processed-data/X_train_processed.csv')
X_valid = pd.read_csv('processed-data/X_valid_processed.csv')
X_test  = pd.read_csv('processed-data/X_test_processed.csv')
y_train = pd.read_csv('processed-data/y_train.csv')
y_valid = pd.read_csv('processed-data/y_valid.csv')
test_id = pd.read_csv('processed-data/test_id.csv') # 這是為了最後上傳用的 ID

# 2. 建立模型 (這裡是可以調整參數的地方 - 報告重點!)
# 改變參數範例: n_estimators (樹的數量), learning_rate (學習率), max_depth (深度)
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=2,          # 從 6 改成 2 (只能做很簡單的判斷)
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

# 3. 訓練模型
print("開始訓練模型...")
# 注意：XGBoost 需要 y 是單一 array，而不是 DataFrame，所以用 .values.ravel()
model.fit(
    X_train, y_train.values.ravel(),
    eval_set=[(X_valid, y_valid.values.ravel())],
    verbose=100  # 每 100 次迭代顯示一次結果
)

# 4. 驗證集評估 (這是寫報告需要的數據)
val_preds = model.predict(X_valid)
acc = accuracy_score(y_valid, val_preds)
print(f"\n驗證集準確率 (Accuracy): {acc:.4f}")
print("分類報告:\n", classification_report(y_valid, val_preds))

# 5. 預測測試集 (這是為了 Kaggle 上傳)
# 注意：測試集沒有 y (因為那是我們要預測的)
test_preds = model.predict(X_test)

# 6. 製作上傳檔案 (Submission File)
# 根據 Kaggle 格式，通常需要 id 和 預測結果
submission = pd.DataFrame({
    'id': test_id['id'],  # 假設 test_id.csv 裡面的欄位叫 'id'
    'Depression': test_preds
})

# 存檔
submission.to_csv('stupid.csv', index=False)
print("\n上傳檔案已建立: submission.csv")