import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "mae": mean_absolute_error(y_true, y_pred)
    }
