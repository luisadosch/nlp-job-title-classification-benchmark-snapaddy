import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def evaluate_predictions(y_true, y_pred):
    metrics = {
        "accuracy": None,
        "macro_f1": None,
        "mae": None
    }

    # Accuracy
    try:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["accuracy"] = None

    # Macro F1
    try:
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    except Exception:
        metrics["macro_f1"] = None

    # MAE
    try:
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
    except Exception:
        metrics["mae"] = None

    return metrics
