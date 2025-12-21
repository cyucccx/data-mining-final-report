import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

def mprint(*args, **kwargs):
    print("[main] ", *args, **kwargs)

def load():
    mprint("Loading processed data...")
    X_train = pd.read_csv('processed-data/X_train_processed.csv')
    X_valid = pd.read_csv('processed-data/X_valid_processed.csv')
    y_train = pd.read_csv('processed-data/y_train.csv')
    y_valid = pd.read_csv('processed-data/y_valid.csv')
    X_test  = pd.read_csv('processed-data/X_test_processed.csv')
    test_id = pd.read_csv('processed-data/test_id.csv')
    return X_train, X_valid, y_train, y_valid, X_test, test_id

def train(X_train, y_train):
    params = {
        'loss': 'log_loss',
        'penalty': 'l2',
        'alpha': 0.0001,
        'max_iter': 1000,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping': True,
        'validation_fraction': 0.1
    }

    model = SGDClassifier(**params)
    model.fit(X_train, y_train.values.ravel())

    return model

def predict(model, X, y=None):
    y_pred = model.predict(X)

    if y is not None:
        report = classification_report(y, y_pred)
        print("Classification Report on Validation Set:")
        print(report)

    return y_pred

def create_submission(test_id, y_test_pred):
    submission = pd.DataFrame({
        'id': test_id['id'],
        'Depression': y_test_pred
    })
    submission.to_csv('submission.csv', index=False)
    mprint("Submission file created: submission.csv")

def main():
    X_train, X_valid, y_train, y_valid, X_test, test_id = load()

    mprint("Training model...")
    clf = train(X_train, y_train)

    mprint("Evaluating on validation set...")
    predict(clf, X_valid, y_valid)

    mprint("Predicting on test set...")
    y_test_pred = predict(clf, X_test)
    create_submission(test_id, y_test_pred)

if __name__ == "__main__":
    main()
