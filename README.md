# Capstone Project – Predicting Career Domain and Seniority from LinkedIn Profiles (Job Titles)

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

## Repository Structure (high level)

- `src/`
  - `prompt_engineering/`  
    Prompt-engineering pipeline using **gemini-2.0-flash** for:
    - evaluation on annotated titles (accuracy/F1 + classification report)
    - synthetic label generation for unlabeled titles (`gemini_synthetic.csv`)
  - `fine_tuning_pretrained/`  
    Transformer fine-tuning experiments using **xlm-roberta-base** for:
    - seniority (regression + classification with synthetic data and oversampling)
    - department (baseline vs oversampling vs synthetic augmentation)
  - `baseline_hybrid_finetuned_approach.ipynb`  
    Hybrid experiment: rule-based first, fine-tuned model as fallback on remaining titles
- `data/`
  - `processed/`  
    Input CSVs (annotated and unlabeled job titles)
  - `results/`  
    Model outputs, including Gemini predictions and synthetic labels

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

## Methods Implemented

### Rule-based matching (baseline)
A deterministic baseline based on keyword/title rules, used as a simple and interpretable reference.

### Simple baselines
- Bag-of-words model
- Embedding-based labeling

### Prompt engineering (Gemini)
We use a system-style prompt with:
- strict closed-set labels
- JSON-only output
- response schema enforcement
- fallback rule for ambiguity

This is used for:
1) direct label prediction on annotated test data  
2) synthetic label generation for unlabeled job titles

**Code:** `src/prompt_engineering/`

### Fine-tuned transformer models 
We fine-tune `xlm-roberta-base` because job titles are multilingual and we observed better OOD behavior than smaller alternatives.

We evaluate:
- Seniority regression baseline (no synthetic data)
- Seniority classification with synthetic data + oversampling
- Department classification:
  - baseline
  - oversampling
  - synthetic data augmentation (best OOD results)

**Code:** `src/fine_tuning_pretrained/`

### Hybrid rule-based + fine-tuned fallback
We test whether rule-based matching can handle easy cases and a fine-tuned model only handles the remainder.

**Code:** `src/baseline_hybrid_finetuned_approach.ipynb`


---
