# Fine-Tuning Models for Seniority and Department Prediction

This repository contains two notebooks that implement and evaluate transformer-based models for predicting **seniority level** and **department** from job titles.

The notebooks combine two complementary approaches and are intentionally separated to ensure efficient experimentation and clear documentation.

Here we tried to combine the two approaches of:
1. Fine-tuning pre-trained transformer models on labeled datasets
2. Generating additional labeled data via prompt engineering for pseudo-labeling
3. Fine-tuning again on the augmented dataset

## Motivation for Separate Notebooks

We split the work into **two notebooks** (one for seniority, one for department) for the following reasons:

- **GPU constraints**:  
  Running both pipelines in a single notebook would require repeated re-training when experimenting with model variants, leading to unnecessary GPU usage.
- **Reproducibility and stability**:  
  Separating notebooks avoids accidental re-runs of expensive training steps.
- **Clear documentation**:  
  Keeping seniority and department modeling separate improves readability and simplifies reporting.

---

## Modeling Approaches

Both notebooks combine the following two approaches:

### 1. Fine-Tuned Classification / Regression Models
We fine-tune a pre-trained transformer model using labeled CSV datasets (`df_seniority` or `df_department`) and then apply the trained model to real LinkedIn CV data.

### 2. Pseudo-Labeling via Prompt Engineering
We generate additional labeled training data by applying prompt-engineered predictions to a large set of previously unlabeled LinkedIn job titles.  
The resulting pseudo-labels are then used to augment the training data and fine-tune the model again.

---

## Data Setup and Evaluation Strategy

### Training Data (In-Distribution)
- `df_seniority` and `df_department` are used as supervised fine-tuning datasets.
- Each dataset is split into **train / validation / test** sets.
  - **Train**: updates model weights
  - **Validation**: used for early stopping and model selection
  - **Test**: provides an in-distribution performance estimate

Early stopping was chosen because it consistently improved model stability and performance.

### Evaluation Data (Out-of-Distribution)
- `jobs_annotated_df` represents real CV data from the production pipeline.
- This dataset is **never used for training or early stopping**.
- It is used exclusively to estimate **out-of-production generalization error** under distribution shift.

---

## Model Choice

For both seniority and department prediction, we use:

**`xlm-roberta-base`**

This model was chosen because:

1. Job titles appear in **multiple languages**
2. Empirical comparison showed better performance than alternatives such as:
   - `distilbert` (≈10% worse accuracy on CV data)
   - `bert-base-cased` (predictions were worse than distillery)

---

## Model Variants and Experiments

Each notebook contains two main experimental setups:

1. **Base model**  
   Fine-tuned only on the provided labeled dataset (`df_seniority` / `df_department`)

2. **Augmented model**  
   Fine-tuned on the labeled dataset plus additional synthetic data generated via prompt engineering

This allows direct comparison between standard fine-tuning and pseudo-label augmentation.

---

## Acquiring Additional Synthetic Data

To increase label coverage and robustness, we generate additional synthetic training data using prompt engineering.

### Motivation
- Labeled data is limited, and we do not have labeled data for the class `Professional`.  With prompt engineering we have tried to acquire this class in our training set.
- Prompt-based labeling outperformed rule-based baselines for both seniority and department
- For department prediction, prompt-engineered labels showed particularly strong improvements
- Therefore we assume the synthetic data is less noisy when using prompt engineering than the rule-based approach

### Data Source for additional Synthetic Data
- For acquiring the additional labels we used the Job titles from the **unlabeled CV JSON dataset**

### Method
- Carefully designed prompts were used to predict seniority and department labels
- The prompt engineering code is provided in the `prompt_engineering/` folder
- Generated labels are treated as pseudo-labels and used only for fine-tuning

---

## Summary

In these notebooks we are combining:
- supervised fine-tuning,
- early stopping with proper validation splits,
- synthetic data generation via prompt engineering,
- and strict out-of-distribution evaluation,

to build robust models for predicting seniority and department from job titles.