from pyexpat import model
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def mprint(*args, **kwargs):
    print("[main] ", *args, **kwargs)

# load data
mprint("Loading processed data...")
X_train = pd.read_csv('processed-data/X_train_processed.csv')
X_valid = pd.read_csv('processed-data/X_valid_processed.csv')
Y_train = pd.read_csv('processed-data/y_train.csv')
Y_valid = pd.read_csv('processed-data/y_valid.csv')
X_test  = pd.read_csv('processed-data/X_test_processed.csv')
test_id = pd.read_csv('processed-data/test_id.csv')

# train model
mprint("Training SVC model...")
svc = SVC(verbose=True)
svc.fit(X_train, Y_train.values.ravel())
print()

# predict on validation set
mprint("Evaluating on validation set...")
Y_valid_pred = svc.predict(X_valid)

mprint("Classification Report on Validation Set:")
report = classification_report(Y_valid, Y_valid_pred)
print(report)

# predict on test set
mprint("Predicting on test set...")
Y_test_pred = svc.predict(X_test)
submission = pd.DataFrame({
    'id': test_id['id'],
    'Depression': Y_test_pred
})
submission.to_csv('submission.csv', index=False)
mprint("Submission file created: submission.csv")
