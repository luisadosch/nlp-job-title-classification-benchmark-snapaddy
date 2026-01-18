# prompt_engineering/gemini_inference.py
import json
import random
import time
from typing import Any, Dict
import streamlit as st
from google import genai

PROMPT_SENIORITY_DEPARTMENT = """
# SCENARIO
You are a **CV expert** specialized in classifying job titles into **seniority levels** and **departments** based on common labor-market and CV conventions.

## TASK
The user will provide **one job title**.
Your task is to classify this job title by:
1. Assigning **exactly one seniority level**
2. Assigning **exactly one department**
3. Providing a **confidence score** between 0.0 and 1.0

Both labels must follow the predefined sets below.

## ALLOWED DEPARTMENT LABELS (Closed Set)
Choose one and only one:
Marketing, Project Management, Administrative, Business Development, Consulting,
Human Resources, Information Technology, Purchasing, Sales, Customer Support, Other

## SENIORITY LABELS (Numeric Mapping)
Choose exactly one:
- Junior -> "1.0"
- Professional -> "2.0"
- Senior -> "3.0"
- Lead -> "4.0"
- Management -> "5.0"
- Director -> "6.0"

## CONFIDENCE
Return "confidence" as a float in [0.0, 1.0].
Guidance: very clear title ~0.8-1.0, somewhat ambiguous ~0.5-0.7, very ambiguous ~0.0-0.4.

## OUTPUT FORMAT
Return ONLY valid JSON that matches the schema.

Fallback Rule:
If ambiguous or no fit: department="Other", seniority_level="2.0", confidence<=0.6

Classify the following job title according to the rules above.
"""

response_schema_seniority_and_department = {
    "type": "object",
    "properties": {
        "seniority_level": {
            "type": "string",
            "enum": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"],
        },
        "department": {
            "type": "string",
            "enum": [
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
            ],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["seniority_level", "department", "confidence"],
}

SENIORITY_MAP = {
    "1.0": "Junior",
    "2.0": "Professional",
    "3.0": "Senior",
    "4.0": "Lead",
    "5.0": "Management",
    "6.0": "Director",
}


@st.cache_resource
def _get_client(api_key: str) -> genai.Client:
    # check if there is a client
    if api_key is None or api_key.strip() == "":
        raise ValueError("Missing Gemini API key, you need to provide a valid key.")

    return genai.Client(api_key=api_key)


def predict_with_gemini(
    job_title: str,
    api_key: str,
    model: str = "gemini-2.0-flash",
    temperature: float = 0.2,
    max_retries: int = 3,
) -> Dict[str, Any]:
    job_title = (job_title or "").strip()
    if not job_title:
        return {"seniority": "—", "department": "—", "confidence": 0.0}

    client = _get_client(api_key)
    full_prompt = PROMPT_SENIORITY_DEPARTMENT + f"\n\nJob title: {job_title}"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": response_schema_seniority_and_department,
                },
            )

            parsed = getattr(resp, "parsed", None)
            if parsed is None:
                text = getattr(resp, "text", "") or ""
                parsed = json.loads(text)

            seniority_level = str(parsed.get("seniority_level", "2.0"))
            department = str(parsed.get("department", "Other"))

            conf_raw = parsed.get("confidence", 0.0)
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            return {
                "seniority": SENIORITY_MAP.get(seniority_level, "Professional"),
                "department": department,
                "confidence": confidence,
                "seniority_level": seniority_level,  # optional für debug
            }

        except Exception as e:
            last_err = e
            sleep = (1.2 * (2 ** (attempt - 1))) + random.random() * 0.4
            time.sleep(sleep)

    return {
        "seniority": "—",
        "department": "—",
        "confidence": 0.0,
        "error": f"Gemini inference failed: {last_err}",
    }
