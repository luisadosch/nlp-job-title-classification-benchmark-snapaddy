# Predict seniority + department for each job title row (row-by-row, save after each row)

import os
import time
import random
import pandas as pd
from google import genai

from prompt_engineering.key import API_KEY
from prompt_engineering.config_utils import (
    PROMPT_SENIORITY_DEPARTMENT,
    response_schema_seniority_and_department
)

class Config:
    def __init__(self):
        self.temperature = 1
        self.prompt = PROMPT_SENIORITY_DEPARTMENT
        self.model = "gemini-2.0-flash"
        self.response_schema = response_schema_seniority_and_department

        self.row_id_column = "row_id"
        self.start_row_id = 0
        self.end_row_id = None

        self.input_csv_path = "../data/processed/jobs_not_annotated.csv"
        self.input_column = "position"

        self.result_csv_path = "../data/results/gemini_synthetic.csv"
        self.result_column_seniority = "seniority"
        self.result_column_department = "department"


class GeminiAPIHandler:
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=API_KEY)

    def get_response(self, full_prompt: str):
        resp = self.client.models.generate_content(
            model=self.config.model,
            contents=full_prompt,
            config={
                "temperature": self.config.temperature,
                "response_mime_type": "application/json",
                "response_schema": self.config.response_schema,
            },
        )
        print("Full response:", resp)
        return resp.parsed  


def run_labeling(cfg: Config, max_retries: int = 3, base_sleep: float = 1.5):
    # you said files exist already -> just load them
    df_in = pd.read_csv(cfg.input_csv_path)
    if "row_id" not in df_in.columns:
        df_in = df_in.reset_index().rename(columns={"index": "row_id"})
        df_in.to_csv(cfg.input_csv_path, index=False)

    df_out = pd.read_csv(cfg.result_csv_path)
    if "row_id" not in df_out.columns:
        df_out = df_out.reset_index().rename(columns={"index": "row_id"})
        df_out.to_csv(cfg.result_csv_path, index=False)     

    handler = GeminiAPIHandler(cfg)

    start = int(cfg.start_row_id)
    end = int(cfg.end_row_id) if cfg.end_row_id is not None else int(df_in[cfg.row_id_column].max()) + 1

    for row_id in range(start, end):
        # find row in input by row_id
        in_mask = df_in[cfg.row_id_column] == row_id
        if not in_mask.any():
            continue
        in_idx = df_in.index[in_mask][0]

        job_title = str(df_in.at[in_idx, cfg.input_column]).strip()
        if not job_title or job_title.lower() == "nan":
            continue

        # find same row in output by row_id
        out_mask = df_out[cfg.row_id_column] == row_id
        if not out_mask.any():
            df_out = pd.concat([df_out, pd.DataFrame([{
                cfg.row_id_column: row_id,
                cfg.input_column: job_title,
                cfg.result_column_seniority: pd.NA,
                cfg.result_column_department: pd.NA,
            }])], ignore_index=True)
            out_idx = df_out.index[-1]
        else:
            out_idx = df_out.index[out_mask][0]

        # skip if already labeled (both fields present)
        if pd.notna(df_out.at[out_idx, cfg.result_column_seniority]) and pd.notna(df_out.at[out_idx, cfg.result_column_department]):
            continue

        full_prompt = cfg.prompt + f"\n\nJob title: {job_title}"

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = handler.get_response(full_prompt)

                seniority_level = result["seniority_level"]   # string like "3.0"
                department = result["department"]             # allowed label

                df_out.at[out_idx, cfg.result_column_seniority] = seniority_level
                df_out.at[out_idx, cfg.result_column_department] = department

                df_out.to_csv(cfg.result_csv_path, index=False)  # save after EACH row
                print(f"row_id={row_id} -> seniority={seniority_level}, department={department}")
                last_err = None
                break

            except Exception as e:
                last_err = e
                sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.5
                print(f"[retry {attempt}/{max_retries}] row_id={row_id} error: {e}")
                time.sleep(sleep)

        if last_err is not None:
            df_out.to_csv(cfg.result_csv_path, index=False)
            raise RuntimeError(f"Stopped at row_id={row_id}. Saved to {cfg.result_csv_path}. Last error: {last_err}")

    return df_out


if __name__ == "__main__":
    cfg = Config()
    cfg.start_row_id = 1000
    cfg.end_row_id = 1887

    labeled_df = run_labeling(cfg)
    print(labeled_df.head(20))
