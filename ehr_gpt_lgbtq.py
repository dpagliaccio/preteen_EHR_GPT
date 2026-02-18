# EHR GPT - LGBTQ+ Identity
# David Pagliaccio
# Feb 17, 2026
# This script runs LLM and RegEx identification of text to find langauge about LGBTQ+ identity

#Load Libraries
from openai import AsyncOpenAI # connects to OpenAI's GPT API
import os #interact with the underlying operating system
import pandas as pd # handles spreadsheet data
import asyncio # enables parallel processing
from tqdm.asyncio import tqdm_asyncio # shows progress bars (optional)
import ehr_gpr_funs as ehr # load internal functions defined in ehr_gpt_funs.py

# time how long script takes to run (optional)
import time
start_time = time.time()  # Start timer
# check max RAM usage (optional)
import psutil
process = psutil.Process(os.getpid())


# Set your OpenAI API key from environment variable
#openai.api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Set system context for the language model (from your helper module)
#system_content = ehr.system_content

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to your input CSV file - output of cleaning step (relative to script location)
input_file = os.path.join(script_dir, "/data/cleaned_output.csv")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Add/initialize required columns to the DataFrame (custom function)
df = ehr.setupdf(df)

# Define the prompt text for the language model (multi-line string)
prompt_text = f"""
You are a clinical language model trained to analyze psychiatric electronic health records from the pediatric emergency room.

Task: Determine if there is any explicit evidence in the following clinical note that the patient identifies as lesbian, gay, bisexual, transgender, queer, intersex, non-binary, or gender-non-conforming. Only base your answer on what is explicitly stated in the note.

**Evidence for a Yes may include:**
- Stated LGBTQ sexual orientation, including 'gay', 'lesbian', 'bisexual', 'homosexual', 'queer', 'LGBTQ',  etc.
- Stated gender identity, including 'transgender', 'trans', 'gender non-conforming', 'gender fluid', 'nonbinary', 'non-binary', etc.
- Uses preferred names or gender neutral pronouns, including she/they, he/they, or they/them pronouns
- Reports stress or issues regarding gender identity or gender dysphoria 
- Explicit reference to dating/relationship history reflecting LGBTQ+ identity (e.g., 'her girlfriend' for a female patient, 'attracted to both men and women')

**Evidence for a No may include:**
- Identifies as heterosexual and cisgender
- Denies any gender identity concerns or denies same-sex attraction


**OUTPUT INSTRUCTIONS:**  
Respond strictly in one of these formats only, with no extra text or formatting:
- Yes | [brief supporting quote from the note]
- No


**EXAMPLES**
- Note: "Patient broke up with his boyfriend and feels depressed"
  Output: Yes | his boyfriend
- Note: "nonbinary child lives with parents and uses they and them pronouns."
  Output: Yes | nonbinary child... uses they and them pronouns
- Note: "Patient denies any sexual or dating history."
  Output: No
- Note: "patient has past medical history lists F64.2 for Gender Identity Disorder"
  Output: Yes | F64.2 for Gender Identity Disorder
- Note: "Patient has both male and female friends."
  Output: No
- Note: "Brother was teased for being gay"
  Output: No
- Note: "said he identifies as bisexual and does not feel accepted after coming out"
  Output: Yes | identifies as bisexual and does not feel accepted after coming out

EHR Note:
"""

# Define the regex pattern
pattern = r"\b(lesbian|gay|bisexual|bisexuality|queer|transgender|asexual|pansexual|homosexual|sexuality|homosexuality|F64|gender identity| gender dysphoria|gender fluid|non[-\s]?binary|gender[-\s]?non[-\s]?conforming|they\s*/\s*them|he\s*/\s*they|she\s*/\s*they|her girlfriend|his boyfriend)\b"

#define function to run gpt prompt on all records
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
    
    tasks = [process_note(idx, row["CleanNote6"]) for idx, row in df.iterrows()]
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Identifying Symptom"):
        result = await coro
        results.append(result)

    for idx, (p1, p2) in results:
        df.at[idx, "Label"] = p1
        df.at[idx, "Supporting_Quote"] = p2


## RUN PARALLEL PROCESSING ON ALL NOTES
asyncio.run(process_all_notes(df, max_concurrent=10))  # 10 parallel processes
# run regex on each row of data & create either/or
for idx, row in df.iterrows():
    note = row["note"]  # Get the clinical note text
    regexyesno, regexmatch = ehr.search_regex(note, pattern)  # Regex search for keywords
    df.at[idx, 'GPT_type'] = ehr.label_type(df.at[idx, 'lgbt'], df.at[idx, 'Label'])  # Compare LLM label to ground truth
    df.at[idx, "Regex"] = regexyesno  # Store regex yes/no
    df.at[idx, "Regex_match"] = regexmatch  # Store regex match string(s)
    df.at[idx, 'Regex_type'] = ehr.label_type(df.at[idx, 'lgbt'], regexyesno)  # Compare regex result to ground truth
    df.at[idx, 'EitherOr_type'] = ehr.label_type_ei(df.at[idx, 'lgbt'], df.at[idx, 'Label'], regexyesno)  # Combined logic

# Calculate F1 and other statistics (custom function)
df = ehr.calcstats(df)

# Save the results to a new CSV file in the data directory
output_file = os.path.join(script_dir, "/data/ehr_LGBTQ_out.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

end_time = time.time()    # End timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")
mem_info = process.memory_info()
print(f"Peak memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")
cpu_percent = process.cpu_percent(interval=1.0)
print(f"CPU usage: {cpu_percent}%")