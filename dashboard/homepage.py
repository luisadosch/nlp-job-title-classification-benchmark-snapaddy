# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from rule_based_inference import predict_rule_based
from bow_inference import predict_bow
from gemini_inference import (
    PROMPT_SENIORITY_DEPARTMENT, predict_with_gemini
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Predicting Career Domain and Seniority from LinkedIn Profiles",
    page_icon="🧭",
    layout="wide",
)

# -----------------------------
# Helpers / placeholders
# -----------------------------
def predict_labels(job_title: str, mode: str) -> dict:
    if not job_title.strip():
        return {"seniority": "—", "department": "—", "confidence": 0.0}

    if mode == "Prompt Engineering":
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return {
                "seniority": "—",
                "department": "—",
                "confidence": 0.0,
                "error": "Missing Streamlit secret GEMINI_API_KEY",
            }

        return predict_with_gemini(job_title=job_title, api_key=api_key)

    # Bag of words approach
    if mode == "Bag of Words":
        out = predict_bow(job_title)
        if not show_debug:
            out.pop("debug", None)
        return out

    if mode == "Rule-based Baseline":
        out = predict_rule_based(job_title)
        if not show_debug:
            out.pop("debug", None)
        return out


    return {"seniority": "—", "department": "—", "confidence": 0.0}



def load_logos():
    uni = Path("images/uni-logo.png")
    snap = Path("images/logo_snapaddy.png")
    return (uni if uni.exists() else None, snap if snap.exists() else None)


# -----------------------------
# Sidebar navigation + settings
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Page",
    ["Prediction Prototype", "Project Overview", "Model Statistics"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.title("Settings")
mode = st.sidebar.selectbox(
    "Inference Mode",
    ["Prompt Engineering", "Bag of Words", "Rule-based Baseline"],
    index=0,
)

st.sidebar.info(
    "This dashboard is a lightweight prototype. Not all project approaches are shown here due to compute/GPU constraints."
)

show_debug = st.sidebar.toggle("Show debug", value=False)

# -----------------------------
# Header: logos side-by-side, then title
# -----------------------------
uni_logo, snap_logo = load_logos()

logo_col1, logo_col2, _ = st.columns([1, 1, 6], vertical_alignment="center")
with logo_col1:
    if uni_logo:
        st.image(str(uni_logo), use_container_width=True)
    else:
        st.caption("⚠️ images/uni-logo.png not found")

with logo_col2:
    if snap_logo:
        st.image(str(snap_logo), use_container_width=True)
    else:
        st.caption("⚠️ images/logo_snapaddy.png not found")

st.markdown(
    """
    <div style="margin-top: 6px;">
      <h1 style="margin-bottom: 4px;">Predicting Career Domain and Seniority from LinkedIn Profiles</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# -----------------------------
# Shared data: metrics table
# -----------------------------

# Summary of model precision metrics on CV set

metrics = pd.DataFrame(
    [
        {"Model variant": "Rule-based baseline", "Target": "Seniority", "Accuracy": 0.3124, "F1": 0.31},
        {"Model variant": "Rule-based baseline", "Target": "Department", "Accuracy": 0.5123, "F1": 0.51},
        {"Model variant": "Prompt engineering (Gemini)", "Target": "Seniority", "Accuracy": 0.5843, "F1": 0.57},
        {"Model variant": "Prompt engineering (Gemini)", "Target": "Department", "Accuracy": 0.7961, "F1": 0.79},
        {"Model variant": "Fine-tuned pretrained transformer model", "Target": "Seniority", "Accuracy": 0.4943, "F1": 0.49},
        {"Model variant": "Fine-tuned pretrained + synthetic", "Target": "Seniority", "Accuracy": 0.6516, "F1": 0.65},
        {"Model variant": "Fine-tuned pretrained transformer model", "Target": "Department", "Accuracy": 0.2792, "F1": 0.27},
        {"Model variant": "Fine-tuned pretrained + synthetic", "Target": "Department", "Accuracy": 0.6886, "F1": 0.69},
    ]
)

# -----------------------------
# Pages
# -----------------------------
if page == "Prediction Prototype":
    st.markdown("## Enter a job title to predict seniority and department")
    st.subheader("🔎 Enter a job title")

    job_title = st.text_input(
        "Job title",
        placeholder='e.g., "Senior Data Scientist" or "Head of Sales DACH"',
    )

    c1, c2, c3 = st.columns([1, 1, 4], vertical_alignment="center")
    with c1:
        run = st.button("Predict", type="primary", use_container_width=True)
    with c2:
        clear = st.button("Reset", use_container_width=True)

    if clear:
        st.session_state.pop("last_prediction", None)
        st.rerun()

    if run:
        st.session_state["last_prediction"] = predict_labels(job_title, mode)
        st.toast("Prediction updated ✅")

    pred = st.session_state.get("last_prediction", {"seniority": "—", "department": "—", "confidence": 0.0})

    st.markdown("### ✅ Prediction output")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.metric("Seniority", pred.get("seniority", "—"))
    with o2:
        st.metric("Department", pred.get("department", "—"))
    with o3:
        conf = float(pred.get("confidence", 0.0))
        st.metric("Confidence", f"{int(conf * 100)}%")

    if mode == "Prompt Engineering":
        st.warning(
            " Prompt Engineering requires a Google Gemini API key.\n\n"
            "Add it in this repo at: `dashboard/.streamlit/secrets.toml`\n\n"
            'Set:\n'
            '`GEMINI_API_KEY = "YOUR_KEY"`',
            icon="⚠️",
        )

    if show_debug:
        st.markdown("### 🧪 Debug")
        st.json({"mode": mode, "job_title": job_title, "prediction": pred})

elif page == "Project Overview":
    st.subheader("📌 Project overview")

    st.markdown(
        """
        This dashboard is a lightweight UI prototype for our capstone project  
        **Predicting Career Domain and Seniority from LinkedIn Profiles**.

        It takes a job title from the **current position** and predicts:

        • **Seniority** (career level)  
        • **Department** (11-class closed set)
        """
    )

    st.markdown("---")

    st.markdown(
        """
        #### Seniority and Department Prediction Challenges
        Real CV titles often look very different from curated training data (**distribution shift**).  
        That’s why we compared multiple approaches instead of relying on a single model.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        #### Approaches implemented in the project
        • **Rule-based matching (baseline):** substring matching against predefined label lists (fast + interpretable).  
        • **Bag of Words (TF–IDF + Logistic Regression):** classical baseline trained on labeled job titles.  
        • **Prompt engineering (Gemini):** few-shot classification using a structured prompt + JSON output schema.  
        • **Fine-tuned transformer:** trained on labeled + synthetic data (not included here due to GPU constraints).  
        • **Embedding-based labeling:** similarity-based matching using embeddings of titles and label descriptions.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        **How to use the dashboard**  
        Pick an inference mode in the sidebar, enter a job title, and click **Predict**.  
        The dashboard returns predicted seniority, department, and a confidence score.
        """
    )

    st.info(
        "⚠️ **Prompt Engineering requires a Gemini API key**\n\n"
        "Add this file in the repo:\n"
        "`dashboard/.streamlit/secrets.toml`\n\n"
        "and set:\n"
        '`GEMINI_API_KEY = "YOUR_KEY"`\n\n'
        "Without a key, Prompt Engineering will not run."
    )



else:  # "Model Statistics"
    st.subheader("📊 Model statistics")
    st.caption("This is a table summarizing precision metrics for our different model variants. The metrics are computed on the out-of-distribution CV dataset.")

    view = st.segmented_control(
        "View",
        options=["All", "Seniority only", "Department only"],
        default="All",
    )

    df = metrics.copy()
    if view == "Seniority only":
        df = df[df["Target"] == "Seniority"]
    elif view == "Department only":
        df = df[df["Target"] == "Department"]

    df_display = df.copy()
    df_display["Accuracy"] = (df_display["Accuracy"] * 100).round(2).astype(str) + "%"
    df_display["F1"] = df_display["F1"].round(2)

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    best = df.sort_values("Accuracy", ascending=False).head(1).iloc[0]
    st.success(f"Highest accuracy: {best['Model variant']} ({best['Target']})")

st.divider()

