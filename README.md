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

## Repository Structure 

- `archive/`
  Task description and meeting protocols.
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
  - `prompt_engineering/`  
    Prompt-engineering pipeline using **gemini-2.0-flash** for:
    - evaluation on annotated titles (accuracy/F1 + classification report)
    - synthetic label generation for unlabeled titles (`gemini_synthetic.csv`)
  
  - `baseline_hybrid_finetuned_approach.ipynb`  
    Hybrid experiment: rule-based first, fine-tuned model as fallback on remaining titles
  - `model-1-baseline.ipynb`  
    rule-based matching baseline approach.
    

### Für dich sonia damit du weißt was das ist
Hinweis für den project overview
in dashboard ordner sind in der folder images die bilder für streamlit dashboard, bow_inference die impleemtnierung für den aufruf des bag of word modells und gemini_inference die implementierung für den aufruf von prompt engineering;
rule_based_inference ist die implementierung für den aufruf des rule based modells (Baseline)
Der hauptcode der streamlit webseite ist in homepage.py (ist nicht im dashboard -> sonst geht deployment nicht - also bitte nicht verschieben)
- im ordner prompt engineering sind config_utils ->  system prompt und output schema für prompt engineering
main processor -> hauptcode für prompt engineering pipeline
prompt_engineering_results.ipynb -> auswertung der ergebnisse von prompt engineering

im ordner fine tuning pretrained sind die ganzen implementierungen für fine tuning von xlm roberta base
- pretrained_class_seniority -> fine tuning für seniority
- pretrained_classification_department -> fine tuning für department 
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

## Results of Methods Implemented

Hier machen wir am ende eine tablle of all results hin

---
