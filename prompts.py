"""summarization prompts.

Based on https://github.com/daveshap/Quickly_Extract_Science_Papers
"""

# Chain of Density
DENSITY_PROMPT = """
Article: {text}

---
Article Summary Guidelines
You will generate increasingly concise, entity-dense summaries of the above article. Repeat the following 2 steps 5 times.

Repeat the following 2 steps 5 times:
Step 1: Identify 1-3 informative entities (";" delimited) from the previous generated version of the summary. 
Step 2: Write a new, denser summary of shorter length that covers every entity mentioned, plus missing entities.

A missing entity is:
- Relevant: to the main story.
- Specific: descriptive yet concise (5 words or fewer).
- Novel: not in the previous summary.
- Faithful: present in the article.
- Anywhere: located anywhere in the article.

Guidelines:
- Make every word count!
- Make space with fusion, compression, and removal of uninformative phrases
- The summaries should become highly dense and concise yet self-contained.
- Missing entities can appear anywhere in the new summary. 
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Answer in JSON. The JSON should be a list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
"""

SUMMARY = (
    "Summarize this text in as much detail as possible. Give a clear explanation of the objectives, core assertions, implications, "
    "and mechanics elucidated in this text - remove citations! \n"
    "Text: {text} \n"
)
