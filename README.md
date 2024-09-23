# pdf_summary_llm
Summarize PDFs from a chosen directory with an LLM. Summaries are saved in a `txt` file.

This application uses Langchain and a HugginFace model. To work properly a `config.py` file is required with the following contents:

```
import os

HUGGINGFACEHUB_API_TOKEN = "YOUR API TOKEN"

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
```
