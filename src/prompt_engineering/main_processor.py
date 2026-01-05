# Problem for finetuning language models -> we have labels in test data, but not in training data
# Solution: create additional training data with labels, use prompt engineering to guide the model to produce the desired output

import os
import time
import random
import pandas as pd
from google import genai
from prompt_engineering.key import API_KEY
from prompt_engineering.config_utils import PROMPT, response_schema

jobs_not_annotated_df = pd.read_csv("../data/processed/jobs_not_annotated.csv")

class Config:
    def __init__(self):
        self.temperature = 0.8
        self.prompt = PROMPT
        self.model =  "gemini-2.0-flash"
        self.response_schema = response_schema

        self.row_id_column = "row_id"
        self.start_row_id = 0
        self.end_row_id = None

        self.input_df = jobs_not_annotated_df
        self.input_column = "position"

        self.result_csv_path = "../data/results/gemini_synthetic.csv"
        self.result_column = "seniority"

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
        return resp.parsed  # <- exactly what you want

def run_labeling(cfg: Config, max_retries=1, base_sleep=1.5):
    # resume from existing csv, else start from input_df
    if os.path.exists(cfg.result_csv_path):
        df = pd.read_csv(cfg.result_csv_path)
    else:
        df = cfg.input_df.copy()
        os.makedirs(os.path.dirname(cfg.result_csv_path), exist_ok=True)
        df.to_csv(cfg.result_csv_path, index=False)

    if cfg.result_column not in df.columns:
        df[cfg.result_column] = pd.NA

    start = int(cfg.start_row_id)
    end = int(cfg.end_row_id) if cfg.end_row_id is not None else int(df[cfg.row_id_column].max()) + 1

    handler = GeminiAPIHandler(cfg)

    for row_id in range(start, end):
        mask = df[cfg.row_id_column] == row_id
        if not mask.any():
            continue
        idx = df.index[mask][0]

        if pd.notna(df.at[idx, cfg.result_column]):
            continue

        job_title = str(df.at[idx, cfg.input_column]).strip()
        if not job_title or job_title.lower() == "nan":
            df.at[idx, cfg.result_column] = pd.NA
            df.to_csv(cfg.result_csv_path, index=False)
            continue

        full_prompt = cfg.prompt + f"\n\nJob title: {job_title}\nReturn only the numeric label."

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result_dict = handler.get_response(full_prompt)  # <- response.parsed
                seniority_level = str(result_dict["seniority_level"])

             
                df.at[idx, cfg.result_column] = seniority_level
                df.to_csv(cfg.result_csv_path, index=False)  # save after each row
                print(f"row_id={row_id} -> {seniority_level}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                sleep = base_sleep * (2 ** (attempt - 1)) + random.random() * 0.5
                print(f"[retry {attempt}/{max_retries}] row_id={row_id} error: {e}")
                time.sleep(sleep)

        if last_err is not None:
            df.to_csv(cfg.result_csv_path, index=False)
            raise RuntimeError(f"Stopped at row_id={row_id}. Saved to {cfg.result_csv_path}. Last error: {last_err}")

    return df

if __name__ == "__main__":
    cfg = Config()
    cfg.start_row_id = 402
    cfg.end_row_id = 420
    labeled_df = run_labeling(cfg)
    print(labeled_df.head(20))
