# bag_of_words/bow_inference.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ALLOWED_DEPARTMENTS = {
    "Marketing",
    "Project Management",
    "Administrative",
    "Business Development",
    "Consulting",
    "Human Resources",
    "Information Technology",
    "Purchasing",
    "Sales",
    "Customer Support",
    "Other",
}

ALLOWED_SENIORITIES = {"Junior", "Professional", "Senior", "Lead", "Management", "Director"}

# “Confidence matching rule”
DEPT_FALLBACK_THRESHOLD = 0.45
SEN_FALLBACK_THRESHOLD = 0.45


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text





@st.cache_data
def _load_training_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    here = Path(__file__).resolve()

    dep_path = here.parent.parent / "data" / "raw" / "department-v2.csv"
    sen_path = here.parent.parent / "data" / "raw" / "seniority-v2.csv"

    dep = pd.read_csv(dep_path)
    sen = pd.read_csv(sen_path)

    if "text" not in dep.columns or "label" not in dep.columns:
        raise ValueError(f"Department CSV must have columns ['text','label'], got: {list(dep.columns)}")
    if "text" not in sen.columns or "label" not in sen.columns:
        raise ValueError(f"Seniority CSV must have columns ['text','label'], got: {list(sen.columns)}")

    dep = dep.dropna(subset=["text", "label"]).copy()
    sen = sen.dropna(subset=["text", "label"]).copy()

    dep["text"] = dep["text"].astype(str).apply(normalize_text)
    dep["label"] = dep["label"].astype(str)

    sen["text"] = sen["text"].astype(str).apply(normalize_text)
    sen["label"] = sen["label"].astype(str)

    return dep, sen


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


@st.cache_resource
def _get_models() -> Tuple[Pipeline, Pipeline, str, str]:
    """
    Train both models once and cache them.
    Returns:
      department_model, seniority_model, dept_majority_label, sen_majority_label
    """
    dep, sen = _load_training_data()

    dmodel = _make_pipeline()
    smodel = _make_pipeline()

    dmodel.fit(dep["text"], dep["label"])
    smodel.fit(sen["text"], sen["label"])

    dept_majority = dep["label"].value_counts().idxmax()
    sen_majority = sen["label"].value_counts().idxmax()

    return dmodel, smodel, str(dept_majority), str(sen_majority)


def _top2(classes: np.ndarray, probs: np.ndarray) -> List[Tuple[str, float]]:
    idx = np.argsort(probs)[::-1][:2]
    return [(str(classes[i]), float(probs[i])) for i in idx]


def predict_bow(job_title: str) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    if not job_title:
        return {"seniority": "—", "department": "—", "confidence": 0.0}

    dmodel, smodel, dept_majority, sen_majority = _get_models()

    x = [normalize_text(job_title)]

    # Department
    d_probs = dmodel.predict_proba(x)[0]
    d_classes = dmodel.named_steps["clf"].classes_
    d_idx = int(np.argmax(d_probs))
    d_label_raw = str(d_classes[d_idx])
    d_conf_raw = float(d_probs[d_idx])

    # Seniority
    s_probs = smodel.predict_proba(x)[0]
    s_classes = smodel.named_steps["clf"].classes_
    s_idx = int(np.argmax(s_probs))
    s_label_raw = str(s_classes[s_idx])
    s_conf_raw = float(s_probs[s_idx])

    # Fallback rule (low-confidence -> default label)
    dept_low = d_conf_raw < DEPT_FALLBACK_THRESHOLD
    sen_low = s_conf_raw < SEN_FALLBACK_THRESHOLD

    dept_final = "Other" if dept_low else d_label_raw
    sen_final = "Professional" if sen_low else s_label_raw

    # If your seniority training set doesn't contain "Professional", keep it anyway for UI consistency.
    # If you prefer majority fallback instead, replace "Professional" with sen_majority.

    # Keep outputs in allowed sets if possible
    if dept_final not in ALLOWED_DEPARTMENTS:
        dept_final = "Other"
        dept_low = True

    if sen_final not in ALLOWED_SENIORITIES:
        sen_final = "Professional"
        sen_low = True

    # Overall confidence: average of both raw confidences + penalty if fallback triggered
    confidence = 0.5 * (d_conf_raw + s_conf_raw)
    if dept_low or sen_low:
        confidence *= 0.6
    confidence = float(max(0.0, min(1.0, confidence)))

    return {
        "seniority": sen_final,
        "department": dept_final,
        "confidence": confidence,
        "debug": {
            "dept_pred_raw": d_label_raw,
            "dept_conf_raw": d_conf_raw,
            "dept_top2": _top2(d_classes, d_probs),
            "dept_fallback": dept_low,
            "dept_majority": dept_majority,
            "sen_pred_raw": s_label_raw,
            "sen_conf_raw": s_conf_raw,
            "sen_top2": _top2(s_classes, s_probs),
            "sen_fallback": sen_low,
            "sen_majority": sen_majority,
            "thresholds": {
                "dept": DEPT_FALLBACK_THRESHOLD,
                "sen": SEN_FALLBACK_THRESHOLD,
            },
        },
    }
