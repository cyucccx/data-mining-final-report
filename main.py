import optuna
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold

def mprint(*args, **kwargs):
    print("[main]", *args, **kwargs)

def load():
    mprint("Loading processed data...")
    X_train = pd.read_csv('processed-data/X_train_processed.csv')
    X_valid = pd.read_csv('processed-data/X_valid_processed.csv')
    y_train = pd.read_csv('processed-data/y_train.csv')
    y_valid = pd.read_csv('processed-data/y_valid.csv')
    X_test  = pd.read_csv('processed-data/X_test_processed.csv')
    test_id = pd.read_csv('processed-data/test_id.csv')
    return X_train, X_valid, y_train, y_valid, X_test, test_id

def objective(trial, X_train, y_train, X_valid, y_valid):
    loss = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'])
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])

    l1_ratio = 0.15
    if penalty == 'elasticnet':
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

    learning_rate = trial.suggest_categorical('learning_rate', ['optimal', 'invscaling', 'adaptive'])
    early_stopping = trial.suggest_categorical('early_stopping', [True, False])
    n_iter_no_change = 5
    if early_stopping:
        n_iter_no_change = trial.suggest_int('n_iter_no_change', 2, 10)

    params = {
        'loss': loss,
        'alpha': alpha,
        'penalty': penalty,
        'l1_ratio': l1_ratio,
        'learning_rate': learning_rate,
        'early_stopping': early_stopping,
        'n_iter_no_change': n_iter_no_change,
    }

    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []
    for train_index, valid_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

        clf = SGDClassifier(**params)
        clf.fit(X_tr, y_tr.values.ravel())

        y_val_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        accuracies.append(accuracy)

    # Return the average accuracy across folds
    return sum(accuracies) / len(accuracies)

def train(X_train, y_train, X_valid, y_valid):
    study = optuna.create_study(direction='maximize', study_name='SGDClassifier Optimization')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    best_params = trial.params
    best_model = SGDClassifier(**best_params)
    best_model.fit(X_train, y_train.values.ravel())

    return best_model

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
    clf = train(X_train, y_train, X_valid, y_valid)

    mprint("Evaluating on validation set...")
    predict(clf, X_valid, y_valid)

    mprint("Predicting on test set...")
    y_test_pred = predict(clf, X_test)
    create_submission(test_id, y_test_pred)

if __name__ == "__main__":
    main()
