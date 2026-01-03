import pandas as pd
def normalize_text(text):
    """
    Minimal text normalization for rule-based baseline.
    This function is intentionally simple and is not used
    by ML-based models.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text
