import pandas as pd

def get_baseline_dataset(df):
    """
    Baseline dataset:
    - uses only the current ACTIVE job title
    - keeps only CVs with exactly one ACTIVE job
    """

    rows = []

    for cv_id, cv_df in df.groupby("cv_id"):
        active_jobs = cv_df[cv_df["status"] == "ACTIVE"]

        # Require exactly one active job
        if len(active_jobs) != 1:
            continue

        current = active_jobs.iloc[0]

        rows.append({
            "cv_id": cv_id,
            "text": current["position"],
            "department": current["department"],
            "seniority": current["seniority"],
        })

    base = pd.DataFrame(rows)

    X = base["text"]
    y_department = base["department"]
    y_seniority = base["seniority"]
    meta = base[["cv_id"]]

    return X, y_department, y_seniority, meta



def get_extended_dataset(df):
    rows = []

    for cv_id, cv_df in df.groupby("cv_id"):

        # Keep only jobs with known temporal meaning
        cv_df = cv_df[cv_df["status"].isin(["ACTIVE", "INACTIVE"])]

        # Sort so that lower job_index = more recent job
        cv_df = cv_df.sort_values("job_index")

        # Identify current job
        active_jobs = cv_df[cv_df["status"] == "ACTIVE"]

        # Require exactly one current job
        if len(active_jobs) != 1:
            continue

        current = active_jobs.iloc[0]

        # Previous jobs = INACTIVE and older than current
        history = cv_df[
            (cv_df["status"] == "INACTIVE") &
            (cv_df["job_index"] > current["job_index"])
        ]

        # Extended text input: current job + previous jobs
        text = current["position"]
        if len(history) > 0:
            text += " | " + " | ".join(history["position"])

        rows.append({
            "cv_id": cv_id,
            "text": text,
            "department": current["department"],
            "seniority": current["seniority"],
        })

    ext = pd.DataFrame(rows)

    X = ext["text"]
    y_department = ext["department"]
    y_seniority = ext["seniority"]
    meta = ext[["cv_id"]]

    return X, y_department, y_seniority, meta
