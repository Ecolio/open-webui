import requests
from fastapi import UploadFile, HTTPException, status
from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
import pandas as pd


def extract_text(file: UploadFile):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext == "pdf":
        loader = PyPDFLoader(file.file)
    elif file_ext in ["txt", "md"]:
        loader = TextLoader(file.file)
    elif file_ext in ["doc", "docx"]:
        loader = Docx2txtLoader(file.file)
    elif file_ext in ["htm", "html"]:
        loader = WebBaseLoader(file.file)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type"
        )

    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    return text


def extract_tables(file: UploadFile):
    # Placeholder for extracting tables from the file
    return []


def process_kpi_tables(tables):
    kpi_summary = {
        "KPI": ["KPI 1", "KPI 2"],
        "Description": ["Description of KPI 1", "Description of KPI 2"],
        "Value": [100, 200],
    }
    return pd.DataFrame(kpi_summary).to_dict(orient="records")


def get_summary_from_ollama(text):
    url = "http://localhost:11434/generate"  # Ollama LLM endpoint
    payload = {"prompt": text, "max_length": 300}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]
