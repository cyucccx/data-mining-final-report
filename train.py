import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import os

# Define paths
DATA_DIR = "processed-data"
X_TRAIN_PATH = os.path.join(DATA_DIR, "X_train_processed.csv")
Y_TRAIN_PATH = os.path.join(DATA_DIR, "y_train.csv")
X_VALID_PATH = os.path.join(DATA_DIR, "X_valid_processed.csv")
Y_VALID_PATH = os.path.join(DATA_DIR, "y_valid.csv")
X_TEST_PATH = os.path.join(DATA_DIR, "X_test_processed.csv")
TEST_ID_PATH = os.path.join(DATA_DIR, "test_id.csv")

def load_data():
    print("Loading data...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    X_valid = pd.read_csv(X_VALID_PATH)
    y_valid = pd.read_csv(Y_VALID_PATH).values.ravel()
    X_test = pd.read_csv(X_TEST_PATH)
    test_id = pd.read_csv(TEST_ID_PATH)
    return X_train, y_train, X_valid, y_valid, X_test, test_id

def train_model(X_train, y_train, X_valid, y_valid):
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(random_state=42)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
    )
    return clf

def evaluate_model(clf, X_valid, y_valid):
    print("Evaluating model...")
    y_pred = clf.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))

def main():
    X_train, y_train, X_valid, y_valid, X_test, test_id = load_data()
    
    clf = train_model(X_train, y_train, X_valid, y_valid)
    
    evaluate_model(clf, X_valid, y_valid)
    
    print("Generating predictions for test set...")
    test_preds = clf.predict(X_test)
    
    submission = pd.DataFrame({
        "id": test_id["id"],
        "Depression": test_preds
    })
    
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
