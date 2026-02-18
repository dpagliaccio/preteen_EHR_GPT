#import openai
import os
#import re
import pandas as pd
import ehr_gpr_funs as ehr
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# time how long script takes to run
import time
start_time = time.time()  # Start timer
# check max RAM usage
import psutil
process = psutil.Process(os.getpid())


# Set your OpenAI API key from environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Set system context for the language model (from your helper module)
#system_content = ehr.system_content

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to your input CSV file (relative to script location)
input_file = os.path.join(script_dir, "../data/all_ehr.csv")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Add/initialize required columns to the DataFrame (custom function)
df = ehr.setupdf(df)

# Define the prompt text for the language model (multi-line string)
prompt_text = f"""
You are a clinical language model trained to analyze psychiatric electronic health records from the pediatric emergency room.

Task: Determine if there is **any explicit evidence** in the following clinical note that the patient has had **one or more prior psychiatric hospitalizations or emergency department visits**.

**Evidence for a Yes may include:**
- Mention of prior psychiatric hospitalization, inpatient (IP) admission, emergency department (ED), or residential treatment facility (RTF) stay. This may include hospitals in the New York area, including NYP, Four Winds, Westchester Medical Center, NYCCC, CHONY, Montefiore, Jacobi, Mt. Sinai, or Bellevue Hospital.
- References like "patient was discharged from...", "history of hospitalization for...", or "previous inpatient admission"
- References to transfer from another inpatient facility, such as "recently transferred from RTF downtown" or "previously admitted to NYP".

**Evidence for a No may include:**
- Explicit denial of prior hospitalization (e.g., "no prior hospitalizations", "no history of hospitalization", "first time recieving psychiatric care")
- Do not consider past suicide attempts or past psychiatric diagnoses as evidence of psychiatric hospitalization.
- Do not consider outpatient therapy or counseling as evidence of hospitalization.
- Do not consider references to medical hospitalizations or surgeries for physical health as evidence of psychiatric hospitalization.
- Do not consider recommendations for future inpatient or psychiatric hospitalization (e.g., "recommend admission to RTF", "patient would benefit from inpatient care") as evidence of past hospitalization.
- Do not consider description of current ED visit (e.g. "being evaluated in ED for...") as evidence of past hospitalization.


**OUTPUT INSTRUCTIONS:**  
Respond strictly in one of these formats only, with no extra text or formatting:
- Yes | [brief supporting quote from the note]
- No

EHR Note:
"""

# Define the regex pattern
pattern = r"\b(residential treatment|residential facility|RTF|intensive day|IDT|prior ED|recent ED|previous ED|prior ER|recent ER|previous ER|psych admissions|psychiatrically hospitalized|psychiatric admissions|psych hospitalizations|psychiatric hospital|psychiatric hospitalizations|prior hospitalizations|recent hospitalizations|previous hospitalizations|NYP Westchester|Westchester Medical|NYPW|4 Winds|Four Winds|WBCH|NYCCCIDT|NYPCW|Bellevue|Bronx Lebanon|CHONY|Montefiore|Sinai|Jacobi)\b"



#async def analyze_symptom_parallel(note, prompt_text):
 #   gptout = await ehr.analyze_symptom_parallel(note, prompt_text)
 #   return gptout

async def process_all_notes(df, max_concurrent):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_note(idx, note):
        async with semaphore:
            try:
                outputs = await ehr.analyze_symptom_parallel(note, prompt_text)
                return idx, outputs
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                return idx, (None, None)
    #print(f"Starting to process {len(df)} notes...")
    
    tasks = [process_note(idx, row["CleanNote6"]) for idx, row in df.iterrows()]
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Identifying Symptom"):
        result = await coro
        results.append(result)
    #tasks = [process_note(idx, row["note"]) for idx, row in df.iterrows()]
    #results = await asyncio.gather(*tasks)

    for idx, (p1, p2) in results:
        df.at[idx, "Label"] = p1
        df.at[idx, "Supporting_Quote"] = p2

asyncio.run(process_all_notes(df, max_concurrent=10))  # parallelism here

# --- Main analysis loop ---
for idx, row in df.iterrows():
    note = row["note"]  # Get the clinical note text
    #label, quote = ehr.analyze_symptom(note, prompt_text)  # Use LLM to analyze note
    regexyesno, regexmatch = ehr.search_regex(note, pattern)  # Regex search for keywords
    #df.at[idx, "Label"] = label  # Store LLM label
    #df.at[idx, "Supporting_Quote"] = quote  # Store LLM supporting quote
    df.at[idx, 'GPT_type'] = ehr.label_type(df.at[idx, 'priorhospitalization'], df.at[idx, 'Label'])  # Compare LLM label to ground truth
    df.at[idx, "Regex"] = regexyesno  # Store regex yes/no
    df.at[idx, "Regex_match"] = regexmatch  # Store regex match string(s)
    df.at[idx, 'Regex_type'] = ehr.label_type(df.at[idx, 'priorhospitalization'], regexyesno)  # Compare regex result to ground truth
    df.at[idx, 'EitherOr_type'] = ehr.label_type_ei(df.at[idx, 'priorhospitalization'], df.at[idx, 'Label'], regexyesno)  # Combined logic

# Calculate F1 and other statistics (custom function)
df = ehr.calcstats(df)

# Save the results to a new CSV file in the data directory
output_file = os.path.join(script_dir, "../data/ehr_priorhosp_analysis_output_CleanNote6_v5gpt4_zeroshot.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

end_time = time.time()    # End timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")
mem_info = process.memory_info()
print(f"Peak memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
cpu_percent = process.cpu_percent(interval=1.0)
print(f"CPU usage: {cpu_percent}%")