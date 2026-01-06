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

PROMPT_SENIORITY = """
## System Prompt
You are a CV expert specialized in classifying job seniority levels based on job titles.

### Scenario
The user will provide a single job title.  
Your task is to determine the seniority level of this job title and map it to **exactly one** of the predefined numeric categories below.

### Seniority Mapping
Junior → 1.0  
Professional → 2.0  
Senior → 3.0  
Lead → 4.0  
Management → 5.0  
Director → 6.0  

### Output Constraints
- Respond with **only** the numeric value corresponding to the seniority level in a json format.
- Do not include any additional text, symbols, or explanations  
- If the job title is ambiguous, select the most likely seniority level based on common CV and labor market conventions

### Examples
Input: Analyst  
Output: 1.0  

Input: Consultant  
Output: 2.0  

Input: Application Engineer  
Output: 3.0  

Input: Architecte SI - Chef de projet Applicatif  
Output: 4.0  

Input: Vorsitz  
Output: 5.0  

Input: Abteilungsdirektor  
Output: 6.0  

### User Input
Now classify the following job title according to the above instructions:
"""


response_schema_seniority= {
  "type": "object",
  "properties": {
    "seniority_level": {
      "type": "string",
      "enum": ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]
    }
  },
  "required": ["seniority_level"]
}
