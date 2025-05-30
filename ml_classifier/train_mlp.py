import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def train_eval(name, X_tr, y_tr, X_vl, y_vl, X_te, y_te):
    clf = MLPClassifier((64, 32), max_iter=300, early_stopping=True, random_state=42)
    clf.fit(X_tr, y_tr)

    def evaluate(X, y):
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else np.full(len(y), 0.5)
        return [
            accuracy_score(y, y_pred),
            precision_score(y, y_pred, zero_division=0),
            recall_score(y, y_pred, zero_division=0),
            f1_score(y, y_pred, zero_division=0),
            roc_auc_score(y, y_prob) if len(np.unique(y)) == 2 else float('nan')
        ]

    return clf, {
        'Train': evaluate(X_tr, y_tr),
        'Val': evaluate(X_vl, y_vl),
        'Test': evaluate(X_te, y_te)
    }