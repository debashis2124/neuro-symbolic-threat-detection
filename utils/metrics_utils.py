import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

def evaluate_model(clf, X, y):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else np.full(len(y), 0.5)
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_prob) if len(np.unique(y)) == 2 else float('nan')
    }

def format_results(metrics_dict):
    return [metrics_dict[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]