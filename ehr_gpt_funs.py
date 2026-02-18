# EHR Helper Functions
# David Pagliaccio
# Feb 17, 2026
# This script defines several general functions to be called in other scripts

#Load Libraries
import openai
import re
import pandas as pd
from openai import AsyncOpenAI
import os
from httpx import Timeout


client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     timeout=Timeout(60))


# PARALLELIZED -- Analyze an EHR note using the language model and return a label and supporting quote
async def analyze_symptom_parallel(ehr_text, prompt_text):
    if pd.isna(ehr_text) or not str(ehr_text).strip():
        return "No", ""  # Return "No" if the note is empty or missing

    # Construct the prompt for the LLM
    prompt = prompt_text + f"\n\"\"\"{ehr_text}\"\"\"\n"
    try:
        response = await client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4o",
            messages=[
                #{"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        # Clean and parse the LLM's output
        raw_output = response.choices[0].message.content.strip()
        cleaned = re.sub(r"[\*\`]", "", raw_output)

        # Split output into label and quote if "|" is present
        if "|" in cleaned:
            label, quote = map(str.strip, cleaned.split("|", 1))
        else:
            label = cleaned.strip()
            quote = ""

        return label, quote

    except Exception as e:
        print(f"Error processing note: {e}")
        return "Error", ""  # Return error label if something goes wrong


# (not parallelized version) Analyze a single EHR note using the language model and return a label and supporting quote
def analyze_symptom(ehr_text, prompt_text):
    if pd.isna(ehr_text) or not str(ehr_text).strip():
        return "No", ""  # Return "No" if the note is empty or missing

    # Construct the prompt for the LLM
    prompt = prompt_text + f"\n\"\"\"{ehr_text}\"\"\"\n"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                #{"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        # Clean and parse the LLM's output
        raw_output = response.choices[0].message.content.strip()
        cleaned = re.sub(r"[\*\`]", "", raw_output)

        # Split output into label and quote if "|" is present
        if "|" in cleaned:
            label, quote = map(str.strip, cleaned.split("|", 1))
        else:
            label = cleaned.strip()
            quote = ""

        return label, quote

    except Exception as e:
        print(f"Error processing note: {e}")
        return "Error", ""  # Return error label if something goes wrong

# Search for keywords/phrases in the note using regex
def search_regex(ehr_text, pattern):
    if pd.isna(ehr_text) or not str(ehr_text).strip():
        return "No", ""
    matches = re.findall(pattern, str(ehr_text), re.IGNORECASE)
    if matches:
        # Remove duplicates, preserve lowercase as found
        matched_str = ", ".join(sorted(set([m.lower() for m in matches])))
        return "Yes", matched_str
    else:
        return "No", ""

# Compare ground truth and predicted label, return classification type
def label_type(val1, val2):
    # TP = True Positive, FN = False Negative, FP = False Positive, TN = True Negative
    if val1.strip().startswith("Y") and val2.strip().startswith("Y"):
        return 'TP'
    elif val1.strip().startswith("Y") and val2.strip().startswith("N"):
        return 'FN'
    elif val1.strip().startswith("N") and val2.strip().startswith("Y"):
        return 'FP'
    elif val1.strip().startswith("N") and val2.strip().startswith("N"):
        return 'TN'
    else:
        return 'ERROR'

# Combined logic for "either/or" classification (LLM or regex positive)
def label_type_ei(val1, val2, val3):
    if val1.strip().startswith("Y") and (val2.strip().startswith("Y") or val3.strip().startswith("Y")):
        return 'TP'
    elif val1.strip().startswith("Y") and (val2.strip().startswith("N") and val3.strip().startswith("N")):
        return 'FN'
    elif val1.strip().startswith("N") and (val2.strip().startswith("Y") or val3.strip().startswith("Y")):
        return 'FP'
    elif val1.strip().startswith("N") and val2.strip().startswith("N") and val3.strip().startswith("N"):
        return 'TN'
    else:
        return 'ERROR'   # In case it doesn't fit any

# Initialize DataFrame columns and set up metric labels
def setupdf(df):
    df["Label"] = ""
    df["Supporting_Quote"] = ""
    df["GPT_type"] = ""
    df["Regex"] = ""
    df["Regex_match"] = ""
    df["Regex_type"] = ""
    df["EitherOr_type"] = ""
    df[" "] = ""
    df["calc"] = ""
    df["calc_Label"] = ""
    df["calc_Regex"] = ""
    df["calc_EitherOr"] = ""
    values = ['True Positive', 'True Negative', 'False Positive', 'False Negative','Specificity', 'SensitivityRecall',  'Precision', 'F1']
    df.iloc[1:9, df.columns.get_loc('calc')] = values
    return df

# Calculate statistics (specificity, precision, recall, F1) and format as percentages
def calcstats(df):
    # Count classification outcomes for each method
    #true positives
    df.at[1, "calc_Label"] = (df['GPT_type'] == 'TP').sum()
    df.at[1, "calc_Regex"] = (df['Regex_type'] == 'TP').sum()
    df.at[1, "calc_EitherOr"] = (df['EitherOr_type'] == 'TP').sum()
    #true negatives
    df.at[2, "calc_Label"] = (df['GPT_type'] == 'TN').sum()
    df.at[2, "calc_Regex"] = (df['Regex_type'] == 'TN').sum()
    df.at[2, "calc_EitherOr"] = (df['EitherOr_type'] == 'TN').sum()
    #False positives
    df.at[3, "calc_Label"] = (df['GPT_type'] == 'FP').sum()
    df.at[3, "calc_Regex"] = (df['Regex_type'] == 'FP').sum()
    df.at[3, "calc_EitherOr"] = (df['EitherOr_type'] == 'FP').sum()
    #false negatives
    df.at[4, "calc_Label"] = (df['GPT_type'] == 'FN').sum()
    df.at[4, "calc_Regex"] = (df['Regex_type'] == 'FN').sum()
    df.at[4, "calc_EitherOr"] = (df['EitherOr_type'] == 'FN').sum()
    #Specificity=(TN 2)/(TN 2 + FP 3)
    df.at[5, "calc_Label"] = df.at[2,"calc_Label"] / (df.at[2,"calc_Label"] + df.at[3,"calc_Label"])
    df.at[5, "calc_Regex"] = df.at[2,"calc_Regex"] / (df.at[2,"calc_Regex"] + df.at[3,"calc_Regex"])
    df.at[5, "calc_EitherOr"] = df.at[2,"calc_EitherOr"] / (df.at[2,"calc_EitherOr"] + df.at[3,"calc_EitherOr"])
    #Sensitivity/Recall=(TP 1)/(TP 1 + FN 4)
    df.at[6, "calc_Label"] = df.at[1,"calc_Label"] / (df.at[1,"calc_Label"] + df.at[4,"calc_Label"])
    df.at[6, "calc_Regex"] = df.at[1,"calc_Regex"] / (df.at[1,"calc_Regex"] + df.at[4,"calc_Regex"])
    df.at[6, "calc_EitherOr"] = df.at[1,"calc_EitherOr"] / (df.at[1,"calc_EitherOr"] + df.at[4,"calc_EitherOr"])
    #Precision = TP 1 / (TP 1 + FP 3)
    df.at[7, "calc_Label"] = df.at[1,"calc_Label"] / (df.at[1,"calc_Label"] + df.at[3,"calc_Label"])
    df.at[7, "calc_Regex"] = df.at[1,"calc_Regex"] / (df.at[1,"calc_Regex"] + df.at[3,"calc_Regex"])
    df.at[7, "calc_EitherOr"] = df.at[1,"calc_EitherOr"] / (df.at[1,"calc_EitherOr"] + df.at[3,"calc_EitherOr"])
    #F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
    df.at[8, "calc_Label"] = 2 * ( df.at[6,"calc_Label"] * df.at[7,"calc_Label"] ) / (df.at[6,"calc_Label"] + df.at[7,"calc_Label"])
    df.at[8, "calc_Regex"] =  2 * ( df.at[6,"calc_Regex"] * df.at[7,"calc_Regex"] ) / (df.at[6,"calc_Regex"] + df.at[7,"calc_Regex"])
    df.at[8, "calc_EitherOr"] =  2 * ( df.at[6,"calc_EitherOr"] * df.at[7,"calc_EitherOr"] ) / (df.at[6,"calc_EitherOr"] + df.at[7,"calc_EitherOr"])

    # Format metrics as percentages with two decimal places
    for row in [5, 6, 7, 8]:
        for col in ["calc_Label", "calc_Regex", "calc_EitherOr"]:
            val = df.at[row, col]
            if pd.notnull(val):
                df.at[row, col] = f"{val:.2%}"
    return df