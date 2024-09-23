import os
from pathlib import Path
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import HuggingFaceHub, PromptTemplate
from langchain.schema import Document

from config import set_environment
from prompts import *

set_environment()
llm = HuggingFaceHub(
    repo_id='tiiuae/falcon-7b-instruct',
    model_kwargs={"temperature": 0.1}
)
llm.client.api_url = 'https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct'


def load_pdf(pdf_path: str) -> list[Document]:
    pdf_loader = PyPDFLoader(pdf_path)
    return pdf_loader.load_and_split()


def summarize_pdf_file(pdf_path: str) -> str:
    # a list of split fragments from the file
    docs = load_pdf(pdf_path)
    # map -> (collapse) -> reduce
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=PromptTemplate(input_variables=["text"], template=SUMMARY),
        #return_map_steps=True
    )

    return chain.run(docs)


def format_summaries(sum_dict: dict) -> ():
    for name, content in sum_dict.items():
        print(name, "\n")
        print(content, "\n")


def summarize_pdfs(directory: str, output_file=False):
    summaries = {}
    if output_file:
        f = open('summary.txt', 'w')
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        # skip non-pdf files
        if not Path(filename).suffix.endswith(".pdf") or not os.path.isfile(full_path):
            continue
        print(f"\nCreating summary for file {filename}")
        summary = summarize_pdf_file(full_path)
        summaries[filename] = summary.strip()
        if output_file:
            f.write(f"\nSummary for {filename}\n")
            f.write(f"{summary.strip()}\n\n")
    if output_file:
        f.close()
    return format_summaries(summaries)


if __name__ == "__main__":
    dir = "pdf"
    summarize_pdfs(dir, True)
