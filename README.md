# Capstone Project – Predicting Career Domain and Seniority from LinkedIn Profiles 

This repository contains our capstone project for predicting **seniority** and **department** labels from job titles extracted from LinkedIn-style profiles / CVs.  
We evaluate multiple approaches (rule-based, classical baselines, prompt engineering, and fine-tuned transformers) with a strict focus on **out-of-distribution (OOD) evaluation** on real CV data.

**Authors:** Dosch Luisa, Bronner Sonia, Hüsam Laura

---

## Project Goal

We solve two labeling tasks from job titles:

1. **Seniority prediction** (ordinal scale mapped to numeric labels **1.0–6.0**)  
2. **Department prediction** (11-class closed set)

A key challenge is **distribution shift**: our curated fine-tuning datasets differ strongly from the real CV dataset (`jobs_annotated_df`), especially regarding label frequencies (e.g., `Professional` for seniority, and `Other` for department).

---
## Data Overview

We use two types of data:

### 1) Curated fine-tuning datasets (in-distribution)
- `df_seniority`: supervised dataset for seniority
- `df_department`: supervised dataset for department
Both are split into **train / validation / test**.  
Validation is used for early stopping and model selection; test provides in-distribution evaluation.

### 2) Real CV dataset (out-of-distribution)
- `jobs_annotated_df`: production-like CV job titles
This dataset is **never used for training or early stopping** and is used only for final OOD evaluation.

---
## Methods

We evaluate multiple approaches:

1. **Rule-based baseline** (keyword/matching logic)
2. **Classical baselines**  
   Bag-of-words and embedding-based classifiers, including augmentation and oversampling variants
3. **Prompt engineering** with **gemini-2.0-flash**  
   Structured output schema, evaluation pipeline, and synthetic label generation
4. **Transformer fine-tuning** with **xlm-roberta-base**  
   Experiments for seniority and department, including synthetic augmentation/oversampling

## Results (OOD on real CV titles)

| Model | Target | Accuracy | Macro F1 | MAE |
|---|---|---:|---:|---:|
| Rule-based | Department | 0.6026 | 0.4492 | – |
| Rule-based | Seniority | 0.5368 | 0.4262 | – |
| Prompt-engineered | Department | 0.7961 | 0.7340 | – |
| Prompt-engineered | Seniority | 0.5843 | 0.5402 | – |
| Transformer-Fine-Tuned | Seniority | 0.4943 | 0.4756 | 0.7751 |
| Transformer-Fine-Tuned-augmented | Seniority | 0.6516 | 0.5840 | – |
| Transformer-Fine-Tuned | Department | 0.2792 | 0.3813 | – |
| Transformer-Fine-Tuned-augmented | Department | 0.6886 | 0.6374 | – |
| Bag-of-words | Seniority | 0.4366 | 0.4093 | – |
| Bag-of-words-augmented | Seniority | 0.6453 | 0.5714 | – |
| Bag-of-words | Department | 0.2231 | 0.3382 | – |
| Bag-of-words-augmented + oversampling | Department | 0.6854 | 0.6119 | – |
| Embedding-based | Seniority | 0.4093 | 0.3504 | – |
| Embedding-based-augmented + oversampling | Seniority | 0.6003 | 0.5271 | – |
| Embedding-based | Department | 0.3146 | 0.3150 | – |
| Embedding-based-augmented + oversampling | Department | 0.6982 | 0.6008 | – |

Notes: Macro F1 is reported for comparability under class imbalance. MAE is only applicable where seniority is treated as regression.
---
## Repository Structure 

- `archive/`
  Task description and meeting protocols.
- `dashboard/`
    - `images/`
      the images for the streamlit dashboard
    - `bow_inference.py`
      dashboard implementation for bag of words models
    - `gemini_inference.py`
      dashboard implementation for prompt engineering
    - `rule_based_inference.py`
      dashboard implementation for the rule-based baseline
- `data/`
  - `raw/`
    annotated csv files that are used for in-distribution training
  - `processed/`  
    Input CSVs (annotated and unlabeled job titles)
  - `results/`  
    Model outputs, including Gemini predictions and synthetic labels
- `src/`
  - - `fine_tuning_pretrained/`  
    Transformer fine-tuning experiments using **xlm-roberta-base** for:
    - seniority (regression + classification with synthetic data and oversampling)
    - department (baseline vs oversampling vs synthetic augmentation)
      - `pretrained_class_seniority.ipynb`
        fine tuning for seniority
      - `pretrained_classification_department.ipynb`
        fine tuning for department 
  - `prompt_engineering/`  
    Prompt-engineering pipeline using **gemini-2.0-flash** for:
    - evaluation on annotated titles (accuracy/F1 + classification report)
    - synthetic label generation for unlabeled titles (`gemini_synthetic.csv`)
      - `config_utils.py`
        system prompt and output schema for prompt engineering
      - `main_processor.py`
        main code for the prompt engineering pipeline
      - `prompt_engineering_results.ipynb`
        evaluation of results of the prompt engineering
  - `utils/`
    utilization functions to call when evaluating model performance and appending model results
      - `eval_utils.py`
        evaluation function to evaluate model on the accuracy_score, f1_score and mean_absolute_error
      - `results_utils.py`
        result function to append model results to the result table
  - `baseline_approach.ipynb`
    rule-based matching baseline approach
  - `baseline_hybrid_finetuned_approach.ipynb`  
    hybrid experiment: rule-based first, fine-tuned model as fallback on remaining titles
  - `bow_approach.ipynb`
    bow approach
  - `embedding-based_approach.ipynb`
    embedding based approach
- `data_prep_eda.ipynb`
  data preparation and preprocessing + exploratory data analysis
- `homepage.py`
  main code for the dashboard
    


---

