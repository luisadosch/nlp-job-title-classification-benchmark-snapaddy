# Fine-Tuning Models for Seniority and Department Prediction

This repository folder contains two notebooks that train and evaluate transformer-based models for predicting **seniority level** and **department** from job titles. The tasks are implemented in separate notebooks to enable efficient experimentation under GPU constraints while keeping the analysis clean and easy to follow.

The overall goal is to study how different training strategies generalize to **real-world CV data**, which differs substantially from the curated fine-tuning datasets.

---

## Modeling Strategy

Across both tasks, we combine three core ideas:

1. **Supervised fine-tuning** of a pre-trained transformer on labeled data  
2. **Synthetic data generation via prompt engineering** to expand label coverage  
3. **Strict out-of-distribution evaluation** on real CV data

This setup allows us to directly compare how well different approaches handle distribution shift.

---

## Data and Evaluation Setup

### In-distribution data
- `df_seniority` and `df_department` are used for supervised fine-tuning.
- Each dataset is split into **train / validation / test**.
  - Train: weight updates  
  - Validation: early stopping and model monitoring  
  - Test: in-distribution performance estimate  

Early stopping is used throughout, as it consistently improved stability and reduced overfitting.

### Out-of-distribution data
- `jobs_annotated_df` represents real CV data from the production pipeline.
- This dataset is never used for training or early stopping.
- It is used exclusively to measure **out-of-production generalization** under distribution shift.

---

## Model Choice

For all experiments we use:

**`xlm-roberta-base`**

Reasons:
- Job titles are multilingual
- In our model runs it outperformed alternatives such as `distilbert` (≈10% worse CV performance)
- More robust under distribution shift than smaller or monolingual models

---

## Experimental Variants

Each notebook evaluates two main variants:

### 1. Fine-tuned (baseline)
The model is fine-tuned only on the original labeled dataset.

### 2. Fine-tuned + synthetic data
The model is fine-tuned on the labeled dataset **augmented with pseudo-labeled job titles** generated via prompt engineering.

This comparison isolates the effect of synthetic data on both in-distribution and out-of-distribution performance.

---

## Synthetic Data via Prompt Engineering

### Motivation
- The original training data is highly imbalanced and poorly aligned with real CV data
- Some important classes are rare or missing entirely (label Professional)
- Prompt-engineered labels consistently outperformed rule-based baselines

### Method
- Unlabeled job titles from the CV JSON data are labeled using carefully designed prompts
- The prompt engineering code and documentation is provided in `prompt_engineering/`
- Generated labels are treated as pseudo-labels and used only for fine-tuning

This approach substantially improves coverage of rare and heterogeneous classes.

---

## Key Findings

### Seniority prediction
- Baseline fine-tuning performs moderately on CV data
- Adding synthetic data improves both accuracy and macro F1
- Gains are especially visible for underrepresented seniority levels

### Department prediction
- Baseline fine-tuning performs very well in-distribution but fails under distribution shift
- Oversampling alone leads to overfitting and worse CV performance
- Synthetic data significantly improves alignment with CV data, especially for the **Other** class
- The augmented model shows the strongest and most stable out-of-distribution performance

---

## Main Results (Out-of-Distribution, CV Data)

| Model variant                         | Target      | Accuracy | Macro F1 | MAE |
|--------------------------------------|-------------|----------|----------|-----|
| Fine-tuned pretrained                | Seniority   | 0.5088   | 0.4421       | 0.7608 |
| Fine-tuned pretrained + synthetic    | Seniority   | **0.6164**   | **0.5586**   | – |
| Fine-tuned pretrained                | Department  | 0.2792   | 0.3813   | – |
| Fine-tuned pretrained + synthetic    | Department  | **0.6886** | **0.6374** | – |

For seniority pretrained, classification metrics are computed after mapping regression outputs back to ordinal levels to ensure comparability. However, we also report MAE as a more natural metric for ordinal regression.

---

## Conclusion

Across both tasks, **synthetic data generated via prompt engineering is the key factor for improving real-world performance**. While pretrained models often achieve near-perfect in-distribution results, they fail under distribution shift. Synthetic augmentation reduces this gap by improving class coverage and better reflecting the structure of real CV job titles.

Overall, the augmented models provide the best trade-off between robustness, generalization, and practical applicability.
