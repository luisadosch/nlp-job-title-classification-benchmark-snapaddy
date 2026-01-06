import pandas as pd
from prompt_engineering.key import API_KEY # key is not uploaded due to privacy reasons
import json
import os
import time
import json
import random
import pandas as pd
from google import genai

# we use these example of training data, as professional I use an eample from myself, since we dont have this exapmle in the training data but do not want to use any test data for training
# we create a prompt that contains these examples and instructs the model to only output the numeric


PROMPT_SENIORITY_DEPARTMENT = """
# SZENARIO
You are a **CV expert** specialized in classifying job titles into **seniority levels** and **departments** based on common labor-market and CV conventions.

## TASK
The user will provide **one job title**.  
Your task is to classify this job title by:
1. Assigning **exactly one seniority level**
2. Assigning **exactly one department**

Both outputs must follow the predefined sets below.

## ALLOWED DEPARTMENT LABELS (Closed Set)

You must choose **one and only one** of the following department labels:

- Marketing  
  Example: *Adjoint directeur communication*

- Project Management  
  Example: *Advisor Strategy and Projects*

- Administrative  
  Example: *Assistent*

- Business Development  
  Example: *Accelerating Business Innovation & Efficiency*

- Consulting  
  Example: *Berater*

- Human Resources  
  Example: *Abteilung HR*

- Information Technology  
  Example: *3rd Level IT Engineer*

- Purchasing  
  Example: *Arbeitsvorbereitung / Technischer Einkauf*

- Sales  
  Example: *Sales Manager B2B/B2C*

- Customer Support  
  Example: *1st Line Support*

- Other  
  Example: *Connected Operations for Industry*

## SENIORITY LABELS (Numeric Mapping)

You must assign **exactly one** of the following seniority levels:

- Junior → **1.0**
- Professional → **2.0**
- Senior → **3.0**
- Lead → **4.0**
- Management → **5.0**
- Director → **6.0**

If the job title is ambiguous, select the **most likely seniority level** based on standard CV and job market conventions.

## Input
The input consists of a **single job title** as plain text.

Examples:
Input: Analyst  
Output:
{
  "department": "Consulting",
  "seniority_level": "1.0"
}

Input: Consultant  
Output: 
{
  "department": "Consulting",
  "seniority_level": "2.0"
}

Input: Application Engineer  
Output: 
{
  "department": "Information Technology",
  "seniority_level": "3.0"
}

Input: Architecte SI - Chef de projet Applicatif  
Output:
{
  "department": "Project Management",
  "seniority_level": "4.0"
}


Input: Vorsitz  
Output: 
{
  "department": "Other",
  "seniority_level": "5.0"
}

Input: Abteilungsdirektor  
Output: 
{
  "department": "Other",
  "seniority_level": "6.0"
}


# RULES
* The output must be valid JSON
* The department must be one of the allowed labels
* The seniority level must be one of: "1.0", "2.0", "3.0", "4.0", "5.0", "6.0"

Task

Classify the following job title according to the rules above.
"""


response_schema_seniority_and_department = {
  "type": "object",
  "properties": {
    "seniority_level": {
      "type": "string",
      "enum": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]
    },
    "department": {
      "type": "string",
      "enum": [
        "Marketing",
        "Project Management",
        "Administrative",
        "Business Development",
        "Consulting",
        "Human Resources",
        "Information Technology",
        "Purchasing",
        "Sales",
        "Customer Support",
        "Other"
      ]
    }
  },
  "required": ["seniority_level", "department"]
}
