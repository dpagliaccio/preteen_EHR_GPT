##### README ##### 

These scripts accompany the journal article "Using Large Language Models to Identify Suicide Risk Factors in Pediatric Emergency Department Chart Notes" in JAACAP Open
Manuscript Authors: David Pagliaccio, Lauren S. Chernick, Ashley Blanchard, Peter S. Dayan, Randy P. Auerbach

Contact Author: David Pagliaccio (david.pagliaccio@nyspi.columbia.edu)


Contents:
ehr_gpt_funs.py -- general python helper functions to support main analyses 
ehr_gpt_cleantext.py -- step 0 cleaning raw EHR notes, 6 sequentially LLM steps
ehr_gpt_lgbtq.py -- identify language indicating lgbtq+ identity LLM + RegEx
ehr_gpt_lgbtq_0shot.py -- modification of above without k-shot examples
ehr_gpt_priorhosp.py -- identify prior hospitalizations LLM + RegEx
ehr_gpt_priorhosp_0shot.py  -- modification of above without k-shot examples
ehr_gpt_sleep.py -- identify sleep problems LLM + RegEx
ehr_gpt_sleep_0shot.py -- modification of above without k-shot examples
bootstrap_metrics.R -- R script to run clustered bootstrapping of performance metrics

Main .py scripts were run with Python3 and call to the ChatGPT Education API
API access (and token key) must be set up before launching these scripts

Inputs are row-wise data with full EHR notes per visit in a single cell 
