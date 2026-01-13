# rule_based_inference.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import streamlit as st

# Seniority mapping (optional, if you want numeric levels too)
SENIORITY_TO_LEVEL = {
    "Junior": "1.0",
    "Professional": "2.0",
    "Senior": "3.0",
    "Lead": "4.0",
    "Management": "5.0",
    "Director": "6.0",
}
LEVEL_TO_SENIORITY = {v: k for k, v in SENIORITY_TO_LEVEL.items()}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = " ".join(text.split())
    return text


def _word_boundary_match(needle_norm: str, haystack_norm: str) -> bool:
    pattern = r"(?:^|\b)" + re.escape(needle_norm) + r"(?:\b|$)"
    return re.search(pattern, haystack_norm) is not None


@dataclass
class MatchResult:
    label: str
    matched_text: Optional[str] = None
    used_fallback: bool = False


@st.cache_data
def load_label_lists(
    department_csv: str = "../data/raw/department-v2.csv",
    seniority_csv: str = "../data/raw/seniority-v2.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dep = pd.read_csv(department_csv)
    sen = pd.read_csv(seniority_csv)

    # basic sanity checks (helps debugging in Streamlit)
    if "text" not in dep.columns or "label" not in dep.columns:
        raise ValueError(f"Department CSV must have columns ['text','label'], got: {list(dep.columns)}")
    if "text" not in sen.columns or "label" not in sen.columns:
        raise ValueError(f"Seniority CSV must have columns ['text','label'], got: {list(sen.columns)}")

    dep["text_norm"] = dep["text"].astype(str).apply(normalize_text)
    sen["text_norm"] = sen["text"].astype(str).apply(normalize_text)

    # prefer longer patterns first (avoid matching "sales" before "field sales")
    dep["__len"] = dep["text_norm"].str.len()
    sen["__len"] = sen["text_norm"].str.len()
    dep = dep.sort_values("__len", ascending=False).drop(columns="__len").reset_index(drop=True)
    sen = sen.sort_values("__len", ascending=False).drop(columns="__len").reset_index(drop=True)

    return dep, sen


def predict_department_rule_based(title: str, department_df: pd.DataFrame) -> MatchResult:
    title_norm = normalize_text(title)

    for _, row in department_df.iterrows():
        needle = row["text_norm"]
        if needle and needle in title_norm:
            return MatchResult(label=row["label"], matched_text=row["text"], used_fallback=False)

    return MatchResult(label="Other", matched_text=None, used_fallback=True)


def predict_seniority_rule_based(title: str, seniority_df: pd.DataFrame) -> MatchResult:
    title_norm = normalize_text(title)

    for _, row in seniority_df.iterrows():
        needle = row["text_norm"]
        if needle and needle in title_norm:
            return MatchResult(label=row["label"], matched_text=row["text"], used_fallback=False)

    # Default fallback if nothing matches
    return MatchResult(label="Professional", matched_text=None, used_fallback=True)


def _match_confidence(title: str, match: MatchResult) -> float:
    """
    Heuristic confidence:
    - fallback => low
    - longer matched phrase => higher
    - word-boundary match => higher (reduces accidental substrings)
    - very long titles => slightly lower
    """
    if match.used_fallback or not match.matched_text:
        return 0.25

    title_norm = normalize_text(title)
    needle_norm = normalize_text(match.matched_text)

    conf = 0.55

    # match specificity
    if len(needle_norm) >= 8:
        conf += 0.25
    elif len(needle_norm) >= 5:
        conf += 0.15
    else:
        conf += 0.05

    # safer match
    if _word_boundary_match(needle_norm, title_norm):
        conf += 0.10

    # noise penalty for very long titles
    if len(title_norm) <= 30:
        conf += 0.05
    elif len(title_norm) >= 80:
        conf -= 0.05

    return max(0.0, min(0.95, conf))


def predict_rule_based(job_title: str) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    if not job_title:
        return {"seniority": "—", "department": "—", "confidence": 0.0}

    dep_df, sen_df = load_label_lists()

    dep_match = predict_department_rule_based(job_title, dep_df)
    sen_match = predict_seniority_rule_based(job_title, sen_df)

    dep_conf = _match_confidence(job_title, dep_match)
    sen_conf = _match_confidence(job_title, sen_match)

    # overall confidence (average of both tasks)
    confidence = max(0.0, min(1.0, 0.5 * (dep_conf + sen_conf)))

    return {
        "seniority": sen_match.label,
        "department": dep_match.label,
        "confidence": confidence,
        # keep for debug; remove in UI if you want
        "debug": {
            "dept_match": dep_match.matched_text,
            "dept_fallback": dep_match.used_fallback,
            "dept_conf": dep_conf,
            "sen_match": sen_match.matched_text,
            "sen_fallback": sen_match.used_fallback,
            "sen_conf": sen_conf,
        },
    }
