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

Task: Determine if there is **any explicit evidence** in the following clinical note for **sleep difficulties**. Only base your answer on what is explicitly stated in the note. 
Respond **Yes** if there is **any mention of a sleep difficulties**, even if other parts of the note contradict or deny it. Prioritize the **presence** of sleep-related concerns over denials or neutral statements.

**Evidence for a Yes may include:**
- Stated symptoms such as insomnia, hypersomnia, somnolence, difficulty falling asleep, frequent waking, excessive sleeping, night terrors, poor sleep, sleep disturbances
- Non-clinical phrases such as "wakes up tired", "not sleeping well", "struggles to sleep", "can't fall asleep"
- Mention of sleep aids, including melatonin
- ICD-10 codes for sleep disorders (e.g., G47)

**Evidence for a No may include:**
- Explicit denial of sleep difficulties (e.g., "denies sleep issues", "reports normal sleep", "no trouble sleeping", "no history of sleep disorder", "no complaints related to sleep")
- Only statements indicating typical or normal sleep with no complaint
- Only references to suicidal or death thoughts (e.g., "wants to fall asleep and never wake up")


**OUTPUT INSTRUCTIONS:**  
Respond strictly in one of these formats only, with no extra text or formatting:
- Yes | [brief supporting quote from the note]
- No

EHR Note:
"""

# Define the regex pattern
pattern = r"\b(insomnia|hypersomnia|somnolent|somnolence|G47|poor sleep|narcolepsy|poor sleep|irregular sleep|sleep aid|melatonin|sleepwake|sleep apnea|sleep disorder|awakening|difficulty sleeping|trouble sleeping|non[-\s]?restorative|sleepiness|night terrors|sleep disturbance|decreased sleep)\b"



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
    df.at[idx, 'GPT_type'] = ehr.label_type(df.at[idx, 'sleep'], df.at[idx, 'Label'])  # Compare LLM label to ground truth
    df.at[idx, "Regex"] = regexyesno  # Store regex yes/no
    df.at[idx, "Regex_match"] = regexmatch  # Store regex match string(s)
    df.at[idx, 'Regex_type'] = ehr.label_type(df.at[idx, 'sleep'], regexyesno)  # Compare regex result to ground truth
    df.at[idx, 'EitherOr_type'] = ehr.label_type_ei(df.at[idx, 'sleep'], df.at[idx, 'Label'], regexyesno)  # Combined logic

# Calculate F1 and other statistics (custom function)
df = ehr.calcstats(df)

# Save the results to a new CSV file in the data directory
output_file = os.path.join(script_dir, "../data/ehr_sleep_analysis_output_CleanNote6_v7gpt4_zeroshot.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

end_time = time.time()    # End timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")
mem_info = process.memory_info()
print(f"Peak memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
cpu_percent = process.cpu_percent(interval=1.0)
print(f"CPU usage: {cpu_percent}%")