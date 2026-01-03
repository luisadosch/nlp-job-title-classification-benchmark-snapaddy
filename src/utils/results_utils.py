import pandas as pd
from pathlib import Path


def add_result(results, model_name, target, extension, metrics):
    """
    Add a single model result to an in-memory results list.

    Parameters
    ----------
    results : list
        List that stores all results for the current notebook.
    model_name : str
        Name of the model (e.g. 'rule_based', 'tfidf_logreg').
    target : str
        Prediction target ('department' or 'seniority').
    extension : bool
        Whether an extension was used.
    metrics : dict
        Dictionary containing evaluation metrics.
        Must include keys: 'accuracy', 'macro_f1'.
    """

    results.append({
        "model": model_name,
        "target": target,
        "extension": extension,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"]
    })


def save_results(results, path="../data/all_results.csv"):
    """
    Persist results to a shared CSV file.
    Creates the file if it does not exist, otherwise appends results.

    Parameters
    ----------
    results : list
        List of result dictionaries.
    path : str
        Path to the CSV results file.
    """

    path = Path(path)
    df = pd.DataFrame(results)

    if path.exists():
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(path, index=False)
