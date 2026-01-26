# app.py
import streamlit as st
import pandas as pd
from dashboard.rule_based_inference import predict_rule_based
from dashboard.bow_inference import predict_bow
from dashboard.gemini_inference import (
    PROMPT_SENIORITY_DEPARTMENT, predict_with_gemini
)
import numpy as np

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
    uni = "dashboard/images/uni-logo.png"
    snap = "dashboard/images/logo_snapaddy.png"
    return uni, snap


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
        {"Model": "Rule-based", "Target": "Department", "Accuracy": 0.6026315789473684, "Macro F1": 0.44918940911600075, "MAE": np.nan},
        {"Model": "Rule-based", "Target": "Seniority",  "Accuracy": 0.5368421052631579, "Macro F1": 0.42616200731397247, "MAE": np.nan},

        {"Model": "Prompt-engineered", "Target": "Department", "Accuracy": 0.7961, "Macro F1": 0.7340, "MAE": np.nan},
        {"Model": "Prompt-engineered", "Target": "Seniority",  "Accuracy": 0.5843, "Macro F1": 0.5402, "MAE": np.nan},

        {"Model": "Transformer-Fine-Tuned", "Target": "Department", "Accuracy": 0.2792, "Macro F1": 0.3813, "MAE": np.nan},
        {"Model": "Transformer-Fine-Tuned-augmented", "Target": "Department", "Accuracy": 0.6886, "Macro F1": 0.6374, "MAE": np.nan},
        {"Model": "Transformer-Fine-Tuned", "Target": "Seniority", "Accuracy": 0.4943, "Macro F1": 0.4756, "MAE": 0.7751},
        {"Model": "Transformer-Fine-Tuned-augmented", "Target": "Seniority", "Accuracy": 0.6516, "Macro F1": 0.5840, "MAE": np.nan},

        {"Model": "Bag-of-words", "Target": "Department", "Accuracy": 0.223114, "Macro F1": 0.338219, "MAE": np.nan},
        {"Model": "Bag-of-words-augmented + oversampling", "Target": "Department", "Accuracy": 0.685393, "Macro F1": 0.611904, "MAE": np.nan},
        {"Model": "Bag-of-words", "Target": "Seniority", "Accuracy": 0.436597, "Macro F1": 0.409319, "MAE": np.nan},
        {"Model": "Bag-of-words-augmented", "Target": "Seniority", "Accuracy": 0.645265, "Macro F1": 0.571373, "MAE": np.nan},

        {"Model": "embedding-based", "Target": "Department", "Accuracy": 0.314607, "Macro F1": 0.314955, "MAE": np.nan},
        {"Model": "embedding-based-augmented + oversampling", "Target": "Department", "Accuracy": 0.698234, "Macro F1": 0.600751, "MAE": np.nan},
        {"Model": "embedding-based", "Target": "Seniority", "Accuracy": 0.409310, "Macro F1": 0.35037, "MAE": np.nan},
        {"Model": "embedding-based-augmented + oversampling", "Target": "Seniority", "Accuracy": 0.600321, "Macro F1": 0.527059, "MAE": np.nan},
    ]
)


# 2) Improvement immer relativ zu Rule-based (pro Target)
# 2) Improvement relativ zu Rule-based (pro Target)
baseline = metrics[metrics["Model"] == "Rule-based"].set_index("Target")["Accuracy"].to_dict()

# ALT (relativ): (acc / baseline - 1) * 100
# NEU (Prozentpunkte):
metrics["Improvement over baseline"] = (
    (metrics["Accuracy"] - metrics["Target"].map(baseline)) * 100
).round(0).astype(int)


# 3) Präsentations-Reihenfolge: immer gleiche Modelle untereinander, erst ohne aug, dann mit aug
def family(m: str) -> str:
    if m == "Rule-based":
        return "Rule-based"
    if m == "Prompt-engineered":
        return "Prompt-engineered"
    if m.startswith("Transformer-Fine-Tuned"):
        return "Transformer-Fine-Tuned"
    if m.startswith("Bag-of-words"):
        return "Bag-of-words"
    if m.startswith("embedding-based"):
        return "embedding-based"
    return m

def is_aug(m: str) -> int:
    return int(("augmented" in m) or ("oversampling" in m))

family_order = {
    "Rule-based": 0,
    "Prompt-engineered": 1,
    "Transformer-Fine-Tuned": 2,
    "Bag-of-words": 3,
    "embedding-based": 4,
}
target_order = {"Department": 0, "Seniority": 1}

metrics["Family"] = metrics["Model"].apply(family)
metrics["Augmented"] = metrics["Model"].apply(is_aug)
metrics["TargetOrder"] = metrics["Target"].map(target_order)
metrics["FamilyOrder"] = metrics["Family"].map(family_order)

metrics = metrics.sort_values(
    ["TargetOrder", "FamilyOrder", "Augmented", "Model"],
    ascending=[True, True, True, True],
).drop(columns=["TargetOrder", "FamilyOrder"])



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



else:  
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

    df_display["Accuracy"] = (df_display["Accuracy"] * 100).round(2).map(lambda x: f"{x:.2f}%")
    df_display["Macro F1"] = (df_display["Macro F1"] * 100).round(2).map(lambda x: f"{x:.2f}%")
    df_display["MAE"] = df_display["MAE"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    def imp_badge(v: int) -> str:
        v = int(v)
        cls = "pos" if v > 0 else ("neg" if v < 0 else "zero")
        sign = "+" if v > 0 else ""
        return f'<span class="imp {cls}">{sign}{v}pp</span>'


    df_display["Improvement over baseline"] = df_display["Improvement over baseline"].apply(imp_badge)

    st.markdown(
        """
        <style>
        .imp {font-weight:800; font-size:1.05rem; padding:2px 10px; border-radius:999px; display:inline-block;}
        .imp.pos {background:#e8f7ee; color:#137a3b; border:1px solid #bfe7cf;}
        .imp.neg {background:#fdecec; color:#b42318; border:1px solid #f3c2c2;}
        .imp.zero{background:#f2f4f7; color:#344054; border:1px solid #d0d5dd;}
        table {width:100%;}
        th, td {padding: 8px 10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_display = df_display.rename(columns={"Model": "Model variant"})
    df_show = df_display[["Model variant", "Target", "Improvement over baseline", "Accuracy", "Macro F1", "MAE"]]

    st.markdown(df_show.to_html(index=False, escape=False), unsafe_allow_html=True)

    best = df.sort_values("Improvement over baseline", ascending=False).iloc[0]
    st.success(
        f"Best improvement: {best['Model']} ({best['Target']}) — "
        f"{best['Improvement over baseline']:+d}pp | "
        f"Acc: {best['Accuracy']*100:.2f}% | "
        f"Macro F1: {best['Macro F1']*100:.2f}%"
    )




st.divider()

