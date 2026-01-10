# README — Prompt Engineering Approach (Gemini)

This folder contains our prompt-engineering pipeline used to **(1) predict labels on annotated CV data** and **(2) generate synthetic labels for additional training data**. We use this approach as a lightweight alternative to training a model from scratch and as a way to improve downstream fine-tuning experiments.

## Goal

We solve two label prediction tasks based on job titles:

* **Seniority prediction** (ordinal, mapped to numeric levels **1.0–6.0**)  
* **Department prediction** (11-class **closed set**)

Prompt engineering is used for two purposes:

1. **Direct prediction on annotated CV job titles (evaluation)**  
   We measure how well the LLM can label real CV job titles with carefully designed instructions and a strict output format.

2. **Synthetic label generation for data augmentation (training)**  
   We apply the same prompting setup to unlabeled job titles to generate additional training samples. These synthetic labels are later used to fine-tune transformer classifier/regressor models.

## Observed prompt-engineering performance (annotated CV dataset)

We evaluated prompt engineering on the annotated CV dataset (`jobs_annotated_active.csv`) and obtained:

* **Seniority Prediction Accuracy:** **58.43%**  
* **Department Prediction Accuracy:** **79.61%**

These results show that department prediction from job titles is substantially easier than seniority prediction, likely because department signals are more explicit (e.g., “Sales”, “IT”, “HR”), while seniority is often ambiguous without additional context (team size, responsibilities, company scale).

## File structure

`prompt_engineering/` contains:

`config_utils.py`  
Configuration for prompt experiments: model settings, temperature, prompt, output schema, and file paths.

`main_processor.py`  
Main script that runs the pipeline: loads job titles, calls Gemini, validates outputs against a JSON schema, and writes results.

Outputs:

`data/results/gemini_results.csv`  
Predictions for the **annotated CV dataset**. Ground truth exists, so we compute accuracy.

`data/results/gemini_synthetic.csv`  
Predictions for **unlabeled job titles** (synthetic training data). No ground truth exists, so we cannot compute accuracy directly. This file is used to augment training data for fine-tuning.

## Prompt design

We use a system-style prompt that enforces:

* **Closed-set** department labels (11 allowed departments)  
* Fixed seniority labels with a **numeric mapping** (1.0–6.0)  
* **Valid JSON output only** (no additional text, to simplify parsing and reduce token usage)  
* A **fallback rule**: if ambiguous, predict `Other` and `2.0` (Professional)

To minimize formatting errors and make downstream processing reliable, we enforce a response schema (`response_schema_seniority_and_department`) that restricts outputs to the allowed values.

This matters because synthetic data is only useful if it is **consistent, machine-readable, and label-clean**.

## Why we used prompt engineering for synthetic data generation

We used prompt engineering instead of only relying on rule-based matching for three reasons:

### 1) Better quality labels than rule-based matching

On the annotated CV dataset, prompt engineering reached **79.61% department accuracy**, which is substantially higher than our rule-based baseline.  
This suggests that the LLM captures semantic signals in job titles (including multilingual ones) better than simple keyword rules.

This is important because synthetic training data needs to be as **low-noise** as possible. If synthetic labels are poor, they can reduce model performance during fine-tuning.

### 2) Handling the missing “Professional” seniority label

A major limitation of rule-based matching is that **“Professional”** is underrepresented in our supervised training set.  
Prompt engineering can still predict this label because the LLM has seen many examples during pretraining and can generalize beyond explicit keywords.

This helps us cover all seniority labels without using test labels for training (no data leakage).

## Interpretation of results

### Seniority (58.43% accuracy)

Seniority is harder because job titles alone often do not encode level reliably. Many errors are “near-misses” on the ordinal scale (e.g., predicting Senior vs Lead).  
We also faced free-tier token limits in Google AI Studio, which restricted systematic prompt iteration.

### Department (79.61% accuracy)

Department is easier because job titles frequently contain strong department cues (e.g., “Sales”, “IT”, “HR”, “Marketing”).  
The high accuracy made prompt engineering a good candidate to generate higher-quality synthetic department labels for fine-tuning.

## How synthetic data is used downstream

`gemini_synthetic.csv` is used to augment the fine-tuning training split:

* We concatenate synthetic `(text, label)` pairs to the supervised training split  
* Validation and test splits remain unchanged to avoid leakage  
* We test whether synthetic augmentation improves **out-of-distribution** performance on CV data

Since synthetic labels have no ground truth, we treat them as a controlled experiment:  
If downstream CV performance improves, the synthetic labels are likely useful; if performance degrades, the synthetic labels may be too noisy or mismatched to the target label definitions.

## Quick start

Run `main_processor.py` to generate predictions. The script reads job titles, applies the prompt, validates outputs via the schema, and writes results to `data/results/`.
