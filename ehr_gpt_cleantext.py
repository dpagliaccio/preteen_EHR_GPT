# EHR Pre-Cleaning
# David Pagliaccio
# Feb 17, 2026
# This script aims to cleans Electronic Health Record (EHR) psychiatric notes by removing boilerplate text, administrative content, and standardized templates while preserving the actual clinical narrative.
# The cleaning is run in 6 serial steps to provide a smaller task within each call
# The API is called using parallel processing to speed up completion time

#Load Libraries
from openai import AsyncOpenAI # connects to OpenAI's GPT API
import os #interact with the underlying operating system
import pandas as pd # handles spreadsheet data
import asyncio # enables parallel processing
from tqdm.asyncio import tqdm_asyncio # shows progress bars (optional)

# time how long script takes to run (optional)
import time
start_time = time.time()  # Start timer
# check max RAM usage (optional)
import psutil
process = psutil.Process(os.getpid())

# Set your OpenAI API key from environment variable ** Make sure OPENAI_API_KEY is set before running this script
# e.g., if launching from bash, can add OPENAI_API_KEY to ~/.bashrc
# alternatively just add the the key in script here
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to your input CSV file (relative to script location)
input_file = os.path.join(script_dir, "/data/input.csv")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(input_file)

# Create 6 new empty columns to store the output from each cleaning phase
df["CleanNote1"] = ""  # After phase 1: headers removed
df["CleanNote2"] = ""  # After phase 2: signatures removed
df["CleanNote3"] = ""  # After phase 3: labs/vitals removed
df["CleanNote4"] = ""  # After phase 4: MSE removed
df["CleanNote5"] = ""  # After phase 5: screening tools removed
df["CleanNote6"] = ""  # After phase 6: final cleanup (duplicates, N/A fields, etc.)

# SYSTEM INSTRUCTIONS - These are the "rules" given to GPT for EVERY API call
# This defines GPT's role and behavior: what it should and shouldn't do
system_content = """
You are a clinical language model designed to clean electronic health record (EHR) notes.

Your task is to remove boilerplate, template, uninformative, and unnecessary content, while strictly preserving the core clinical narrative.

DO NOT:
- Rewrite, paraphrase, summarize, duplicate, or reorder the text.
- Change formatting or structure
- Add any content or assumptions

DO:
- Remove text that matches or closely resembles the task examples
- Use semantic understanding to remove content even when it is phrased differently
- Apply each removal task separately and completely, scanning for all similar matches
- After applying all tasks, re-check the note to ensure no examples remain.
- Leave the note unchanged if no matching content is found

Return only the cleaned note as plain text. Do not include explanations, tags, or metadata.

"""

# PHASE 1 PROMPT - Removes administrative headers and boilerplate
# This phase targets document headers, telehealth disclaimers, interpreter statements, and contact tables
phase_1_prompt = """
**Tasks**
1. **Remove boilerplate headers at the top of the note**. Examples include:
    - "PSY CPEP ED INITIAL EVALUATION NOTE IP NYC"
    - "Ped Psych. Initial Psychiatric Evaluation"
    - "PEDIATRIC PSYCHIATRY ED INITIAL EVALUATION NOTE"

2. **Remove all Telehealth disclaimer or setup language**. Examples include:
    - "This visit was provided by a secure Telehealth system. The patient/parent/guardian is aware of their right to refuse to participate in services delivered via telemedicine and the alternatives and potential limitations of participating in a telemedicine visit versus a face-to-face visit"
    - "I have also informed the patient/parent/guardian of my current location and the names of all persons participating in the telemedicine service and their role in the encounter. The patient/parent/guardian agrees to have this service via Telehealth.
    - "chart assessment of patient done using Avizia cart for telepsychiatry"

3. **Remove interpreter usage statements**. Examples include:
    - "Interpreter used? No"
    - "Interpreter: Interpreter Used? not applicable"
    - "Interpreter: Interpreter Used? Yes. Language: Spanish. Interpreter's Name: #123456"

4. **Remove structured administrative tables or contact info lines**. Examples include:
    - "Psychiatric/Medical/Physical Care: Provider Name Relations Phone Comments"
    - "Care Team Provider Name Relationship Specialty Jane Smith PCP - General Provider Name Relations Phone Comments John Jones, MD Psychiatrist" 
    - "Social Support: Name Relations Phone Comments"

*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*

Note:
""" 

# PHASE 2 PROMPT - Removes EHR system documentation and signatures
# This phase targets note-sharing instructions, attestation statements, and clinician signatures
phase_2_prompt = """
**Tasks**
1. **Remove template-like phrases about EHR documentation, note sharing, and attestation**. Examples include:
    - "No need to delete this statement. It will not file to the chart."
    - "Most notes should be released to patients and their proxies via NYP Connect."
    - "Allowable exceptions include, 1 if you believe there is substantial risk of patient harm, 2 to protect patient privacy, and 3 other permissible reasons such as protecting psychiatric information under the New York State law or protecting information for an active court case."
    - "To BLOCK the release of this note, UNCLICK the Share w/ Patient button in the upper right corner."
    - "Notes for patients < 18 years old, Behavioral Health BH notes, and notes written by non-providers anyone who is not an MD, PA or NP will NOT be automatically released to the patient EVEN IF the Share w/ Patient button is clicked and active."
    - "If a patient or proxy asks you for a note on NYP Connect that you cannot share via Epic, please directly them to Health Information Management HIM to have the note loaded to Connect."
    - "Attestation Statement I am the attending physician for [patient name] and have seen the patient face to face. I personally performed a history and physical exam and discussed the medical decision making with the resident/fellow."
    - "I have made my decisions with the support of the information provided to me by the patient, family, resident/fellow. I agree with the plan as documented by the resident/fellow"
    - "I was wearing the required PPE including appropriate N95/Surgical Mask, Eye Protection, Gown and Gloves for any Aerosol-Generating Procedures if indicated, and I did not have > 15 minutes of unprotected exposure within 6 feet of a SARS-CoV-2 positive patient who did not require an Aerosol-Generating Procedure"
    - "Please select SmartPhrases below to use"
    
2. **Remove clinician names, credentials, and electronic signature blocks**. Examples include:
    - "Completed by: Jane Smith, MD Resident 01/21/21", "COMPLETED BY : [Clinician Name]"
    - "Signed electronically by Jane Smith, MD"
    - "This note was electronically signed and authorized by [Clinician Name] on [Date] at [Time]."
    - "I, [Clinician Name], have personally reviewed this report and concur with its findings and conclusions, as affirmed by my electronic signature"
    

*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*
    
Note:
""" 
# PHASE 3 PROMPT - Removes lab results, Review of Systems, physical exams, and vital signs
# This phase cleans out all the technical medical data that's structured/templated
phase_3_prompt = """
**Tasks**
1. **Remove any lab results**, including full panels, individual result values, and technical or abbreviated listings. Examples include:
    - "Labs Reviewed CBC W/ PLATELETS"
    - "Lab results: Labs Reviewed CBC W/ PLATELETS DIFF (COMPLETE) - Abnormal Result Value White Blood Cell 6.70"
    - "HEPATIC FUNCTION PANEL - Abnormal Protein Total - Abnormal Protein Total 7.8" 
    - "URINALYSIS, COMPLETE W/REFLEX CULTURE - Abnormal Urine Color Yellow Urine Urine Specific Gravity 1.018"
    - "DRUG SCREEN, URINE W/O CONFIRM U Amph/Meth Scr Negative" 
    - "SARS-COV-2 NAAT SARS-CoV-2 NAAT Not Detected"
    - "URINE W/O CONFIRM HEPATIC FUNCTION PANEL ETHANOL SALICYLATE TSH W REFLEX TO FT4 URINALYSIS, DIPSTICK WITH MICROSCOPIC EXAM ON POSITIVES URINALYSIS MICROSCOPIC URINE CULTURE URINALYSIS POCT"
    - "Result Value Urine Bilirubin POC Negative Urine Blood POC Negative Urine Glucose POC Negative Urine Ketones POC Negative Urine Leukocyte Esterase POC Negative Urine Nitrites POC"

2. **Remove Review of Systems (ROS) and Physical Exam sections**, including structured lists. Examples include:
    - "Review of Systems Review of Systems Constitutional: Negative for activity change, appetite change and fever. HENT: Negative for congestion. Respiratory: Negative for cough. Gastrointestinal: Negative for abdominal pain, diarrhea and vomiting. Skin: Negative for rash and wound."
    - "ROS Physical Exam  Constitutional: General: active and not in acute distress. HENT: Head: Normocephalic and atraumatic. Right Ear: Tympanic membrane normal. Left Ear: Tympanic membrane normal. Nose: Nose normal. No congestion or rhinorrhea. Mouth/Throat: Mouth: Mucous membranes are moist. Pharynx: No oropharyngeal exudate or posterior oropharyngeal erythema. Eyes: General: Right eye: No discharge. Left eye: No discharge. Extraocular Movements: Extraocular movements intact. Conjunctiva/sclera: Conjunctivae normal. Pupils: Pupils are equal, round, and reactive to light." 
    - "Physical Examination CONSTITUTIONAL: GENERAL: pt is Alert, Active, in no acute distress APPEARANCE: Normal appearance. EYES: PERL, EOM intact, Conjunctiva normal HENT: Head: Normocephalic."
    
3. **Remove Vital Signs (VS)** sections, including any standard vital measurements. Examples include:
    - "Vitals: 01/01/23 BP: 100/70 Pulse: 70 Resp: 19 Temp: 37.5°C TempSrc: Oral SpO2: 100% Weight: 35.0 kg Physical Exam Vitals and nursing note reviewed."

*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*

Note:
""" 

# PHASE 4 PROMPT - Removes structured Mental Status Exam (MSE) blocks
# IMPORTANT: Only removes STRUCTURED MSE (with labels like "Appearance:", "Mood:", etc.)
phase_4_prompt = f"""
**Task**
**Remove Mental Status Exam (MSE)**
    - This structured block includes a list of labeled fields. Examples include: "Appearance:", "Hygiene/Grooming:", "Dress:", "Attitude:", "Relatedness:", "Eye Contact:", "Attention:", "Psychomotor/Station/Gait:", "Muscle Strength and Tone:", "Abnormal Movements:", "Speech:", "Language:", "Rate:", "Rhythm/Prosody:", "Volume:", "Mood:", "Affect:", "Congruent and Broad Thought Process:", "Goal-directed Thought Content: Suicidal Ideation:",  "Homicidal Ideation:", "Perceptions-Hallucinations:", "Insight:", "Judgement:", "Impulse Control:", "Orientation: Time, Place and Person Level of Consciousness:", "Cognitive Functioning/Memory:"
    - Each labeled field may have a brief description. Examples include: "Denies", "Normal", "Euthymic and alert", or "Grossly intact" after it
    - Example: "Mental Status Exam: Appearance: Stated age Hygiene/Grooming: Well-groomed. Dress: Appropriate Attitude: Patient is Cooperative. Relatedness: Eye Contact: Good Attention: Attentive Psychomotor/Station/Gait: Normokinetic Muscle Strength and Tone: Normal Abnormal Movements: None Speech: Spontaneous and Fluent Language: Grossly intact Rate: Normal Rhythm/Prosody: Normal Volume: Normal Mood: sad per pt Affect: Euthymic, Nonlabile, Congruent and Broad Thought Process: Goal-directed Thought Content: Suicidal Ideation: Denies Homicidal Ideation: Denies Perceptions-Hallucinations: Denies Insight: Good Judgement: Good Impulse Control: Appropriate Orientation: Time, Place and Person Level of Consciousness: Alert Cognitive Functioning/Memory: Grossly intact"

**Do NOT** remove MSE information if it is not structured with labels. Examples include: "On MSE, patient appears older than stated age, well-groomed, and dressed appropriately."


*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*

Note:
"""

# PHASE 5 PROMPT - Removes screening tools and safety plans
# This phase targets CRAFFT substance use screening, C-SSRS/SAFE-T suicide assessments, and safety plan templates 
# This data can be otherwise extracted as structured data
phase_5_prompt = f"""
**Tasks**
1. **Remove substance use screening (CRAFFT)**. Examples include:
    - "CRAFFT: CRAFFT Flowsheet Row Most Recent Value CRAFFT Part A CRAFFT Substance Use Tool Able to complete tool at this time 1. During the PAST 12 MONTHS, on how many days did you drink more than a few sips of beer, wine, or any drink containing alcohol? 0 2. During the PAST 12 MONTHS, on how many days did you use any marijuana (cannabis, weed, oil, wax, or hash by smoking, vaping, dabbing, or in edibles) or "synthetic marijuana" (like "K2," "Spice")? 0 3. During the PAST 12 MONTHS, on how many days did you use anything else to get high (like other illegal drugs, pills, prescription or over-the-counter medications, and things that you sniff, huff, vape, or inject)? 0 CRAFFT SubTotal: 0 4. During the PAST 12 MONTHS, on how many days did you use a vaping device containing nicotine or flavors, or use any tobacco products? 0 CRAFFT Part B (FOR PREGNANT ADOLESCENTS, ANY REPORTED SUBSTANCE USE SHOULD BE CONSIDERED HIGH RISK BEHAVIOR) C: Have you ever ridden in a CAR driven by someone (including yourself) who was "high" or had been using alcohol or drugs? No CRAFFT Total points: (If score is below 2 no intervention is needed. If score is 2 or above provide intervention.) 0 CRAFFT Part B Substance Use Risk Level (FOR PREGNANT ADOLESCENTS, ANY REPORTED SUBSTANCE USE SHOULD BE CONSIDERED HIGH RISK BEHAVIOR) Low Risk (No use in the past 12 months AND score of 0) CRAFFT Part C - The following questions ask about your use of any vaping devices containing nicotine CRAFFT Part C Substance Use Risk Level (FOR PREGNANT ADOLESCENTS, ANY REPORTED SUBSTANCE USE SHOULD BE CONSIDERED HIGH RISK BEHAVIOR) No to Low Risk (0 or 1 day of vaping or tobacco use) Substance Use Plan No to Low Risk Provide information about risks of nicotine use, Offer praise and encouragement"
    - "No Substance Use History/Treatment/Withdraw/Complications: Denies Substance Use Screening: Substance Use"

2. **Remove C-SSRS and SAFE-T results**. Examples include:
    - "C-SSRS Suicide Risk Screening Tool: Able to complete screening tool at this time"
    - "CSSRS 1. Have you wished you were dead or wished you could go to sleep and not wake up?: No 2. Have you actually had any thoughts of killing yourself?: No 6. Have you ever done anything, started to do anything, or prepared to do anything to end your life (LIFETIME OR SINCE LAST VISIT/ASSESSMENT)?: No"
    - "Was a Suicide Risk Assessment (SAFE-T) completed? Yes SAFE-T Assessment (please complete in the flowsheet section by searching for SAFE-T): SAFE-T Suicide Risk Assessment SAFE - T Assessment Completed: yes Suicidal Behavior: History of suicide attempts or self injurious behavior: no" 

3. **Remove Safety Plan template text**. Examples include:
    - "A Safety Plan is a list of coping strategies and supports that you can use when you are in a crisis or very distressed. The 7 steps in the plan can be followed in order to help prevent danger and/or a crisis. Participants in the Safety Plan: pt, mother Target Behaviors: Targeted Behaviors: Non-Suicidal Self-Injurious Behaviors The things that motivate me towards recovery are: my family Step 1: Making the environment safe: limit access to sharp objects 2. keep doors secured 3. Step 2: Triggers: 1. Thinking about negative things 2. Anxiety 3. Step 3: Warning signs: 1. body shaking 2. Breathing heavily 3. Feeling overwhelmed Step 4: Coping strategies: 1. Breathing  2. games 5: People and social settings that help you take your mind off your problems and focus on other things: 1. Name and phone number:  2. Name and phone number: 3. Place:  4. Place:  Step 6: People whom I can ask for help: 1. Name and phone number: 2. Name and phone number: 3. Name and phone number: Step 7: Professionals or agencies I can call during a crisis: CALL 911 IF THERE IS AN EMERGENCY 1. Clinician Name and Phone Number: Clinician Pager or Emergency Contact #: 2. Clinician Name and Phone Number: Clinician Pager or Emergency Contact #: 3. Local Urgent Care Services: MSCH ED 4. Local Urgent Care Services: 988 National Suicide Prevention Hotline: National suicide prevention hotline: Dial "988" or 1-800-273-TALK (1-800-273-8255) - Available 24/7 Location of Crisis Services: Crisis Location: New York City: 1-888-NYC-WELL (1-888-692-9365) (Spanish, Mandarin Language prompts available) Patient participated in, agreed with, and was provided a copy of this plan: Yes Patient Parent and other support signature: (as applicable, optional) Caregiver: Discussed with patient/family/caregiver"

*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*

Note:
"""

# PHASE 6 PROMPT - Final cleanup pass
# Removes discharge plans, allergies, firearm access questions, N/A fields, and duplicate text
phase_6_prompt = f"""
**Tasks**
1. **Remove Discharge plans or Disposition summaries**. Examples include:
    - "Disposition: Discharge home with plan to see outpatient mental health professional"
    - "Disposition Interventions: Should be mitigated by referring back to outpatient provider"
    - "Plan: Patient is safe to return to home"
    - "Plan: Admit to inpatient psychiatric unit. Legal signed by mother. Bed search in progress."

2. **Remove Allergy statements**. Examples include:
    - "Allergies: NKDA"
    - "Allergies: Review of patient's allergies indicates no active allergies."
    - "Allergies: Grass, Dogs, Penicillin"

3. **Remove access to firearms review**. Examples include:
    - "Treatment plan for firearms access: No"
    - "Access to Firearms: No"
    - "Firearms Does the patient have access to firearms? No Any change in access to firearms in the last month? No"

4. **Remove Headers or responses with only "N/A", "Not on file", "No data to display", “Not applicable”**. Examples include:
    - "Substance use history: N/A"
    - "Family History: Not on file"
    - "Additional Comments: N/A"
    - "Radiology Results: No orders to display"
    - "Past Medical History No past medical history documented"
    - "Pertinent Negatives No pertinent negatives documented."

5. **Remove duplicated text** — if a sentence or phrase appears more than once, **remove all but one copy**.


*NEVER rewrite, paraphrase, summarize, duplicate, or reorder the text*

Note:
"""

# FUNCTION: call_gpt - Makes a single API call to GPT-4o
# Parameters:
#   - prompt: The user message (phase prompt + note text)
#   - system_content: The system instructions (how GPT should behave)
# Returns: The cleaned text from GPT
# Note: async allows this function to run in parallel with other API calls
# SET MODEL TO GPT 4o
# SET TEMPERATURE TO 0 (default=1)
async def call_gpt(prompt, system_content):
    # Make async API call to OpenAI's GPT-4o model
    response = await client.chat.completions.create(
        model="gpt-4o",  # Using GPT-4o (can change to other GPT options)
        messages=[{"role": "system", "content": system_content},  # The rules/behavior
                  {"role": "user", "content": prompt}],            # The task + note
        temperature=0  # Deterministic output (same input always gives same output)
    )
    # Extract the text from the response and remove leading/trailing whitespace
    return response.choices[0].message.content.strip()

# FUNCTION: multi_step_clean - Processes a single note through all 6 phases sequentially
# Parameter: note - The original raw EHR note text
# Returns: 6 outputs (one from each phase)
# Each phase takes the output of the previous phase as input
async def multi_step_clean(note):
    phase_1_output = await call_gpt(phase_1_prompt + note, system_content)
    phase_2_output = await call_gpt(phase_2_prompt + phase_1_output, system_content)
    phase_3_output = await call_gpt(phase_3_prompt + phase_2_output, system_content)
    phase_4_output = await call_gpt(phase_4_prompt + phase_3_output, system_content)
    phase_5_output = await call_gpt(phase_5_prompt + phase_4_output, system_content)
    phase_6_output = await call_gpt(phase_6_prompt + phase_5_output, system_content)
    return phase_1_output,phase_2_output,phase_3_output,phase_4_output,phase_5_output, phase_6_output

# FUNCTION: process_all_notes - Processes all notes in parallel with concurrency control
# Parameters:
#   - df: The pandas DataFrame containing all notes
#   - max_concurrent: Maximum number of simultaneous API calls (default 10 - can increase depending on capacity/need)
# This function enables parallel processing while respecting API rate limits
async def process_all_notes(df, max_concurrent):
    # Semaphore limits how many API calls can run simultaneously
    semaphore = asyncio.Semaphore(max_concurrent)

    # Inner function to process a single note with semaphore control
    async def process_note(idx, note):
        async with semaphore:  # Wait for a token, then proceed
            try:
                # Run all 6 phases sequentially on this note
                outputs = await multi_step_clean(note)
                return idx, outputs  # Return the row index and all 6 outputs
            except Exception as e:
                # If this note fails, print error but don't crash the whole script
                print(f"Error at row {idx}: {e}")
                return idx, (None, None, None, None, None, None)  # Return None for failed note
    # Create a task (async job) for each note in the dataframe
    # This starts all the cleaning processes, but limited by the semaphore
    # SET "note" as the name of the name of the input column for the raw data
    tasks = [process_note(idx, row["note"]) for idx, row in df.iterrows()]
    
    # Process tasks as they complete and show progress bar
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Cleaning Notes"):
        result = await coro  # Wait for this task to finish
        results.append(result)  # Collect the result

    # Write all the results back to the dataframe
    # Loop through results and assign to the appropriate row and column
    for idx, (p1, p2, p3, p4, p5, p6) in results:
        df.at[idx, "CleanNote1"] = p1  # Phase 1 output
        df.at[idx, "CleanNote2"] = p2  # Phase 2 output
        df.at[idx, "CleanNote3"] = p3  # Phase 3 output
        df.at[idx, "CleanNote4"] = p4  # Phase 4 output
        df.at[idx, "CleanNote5"] = p5  # Phase 5 output
        df.at[idx, "CleanNote6"] = p6  # Final cleaned note


# ============================================================================
# MAIN EXECUTION - Run the cleaning process
# ============================================================================

# RUN THE PROCESSING
# asyncio.run() executes the async function and waits for all notes to complete
# max_concurrent=10 means process up to 10 notes simultaneously 
asyncio.run(process_all_notes(df, max_concurrent=10))  

# Save the results to a new CSV file in the data directory
output_file = os.path.join(script_dir, "/data/cleaned_output.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")

# PERFORMANCE METRICS
end_time = time.time()    # End timer
print(f"Script runtime: {end_time - start_time:.2f} seconds")
mem_info = process.memory_info()
print(f"Peak memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")