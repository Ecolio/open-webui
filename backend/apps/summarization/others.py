from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index import QueryEngine, MetadataFilters
import requests
import json

# important information regarding the llama3 extented context window model: https://ollama.com/library/llama3-gradient

def vector_query_tool(
    query: str, page_numbers: List[str] = [], document: str = None
) -> str:
    """Perform a vector search over an index.

    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave EMPTY if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    document (str): the path to the document. If provided, the text will be extracted from the document.

    """
    # Extract the text from the document, if provided
    if document:
        text = extract_text(document)
    else:
        text = ""

    # Initialize the query engine
    query_engine = QueryEngine(vector_store="faiss")

    # Load the index
    query_engine.load_index("path/to/your/index")

    # Define the metadata filters
    if page_numbers:
        metadata_filters = MetadataFilters.from_dicts(
            [{"key": "page_label", "value": p} for p in page_numbers],
            condition=FilterCondition.OR,
        )
    else:
        metadata_filters = None

    # Perform the search
    response = query_engine.query(
        query,
        similarity_top_k=2,
        filters=metadata_filters,
    )

    # Extract the text from the response
    response_text = "\n\n".join([n.get_content() for n in response.source_nodes])

    # Generate the text using the local Llama model
    data = {
        "text": f"{text}\n\n{response_text}\n\nAnswer the following question:\n\n{query}",
        "top_k": 1,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        "http://localhost:11434/generate", data=json.dumps(data), headers=headers
    )
    generated_text = response.json()["text"]

    return generated_text


import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt_summarizer import GPT3Summarizer
import requests
import json


def summary_tool(
    text: str, max_length: int = 100, model_name: str = "gpt2", document: str = None
) -> str:
    """Provide a summary of the given text.

    text (str): the text to be summarized.
    max_length (int): the maximum length of the summary.
    model_name (str): the model to use for summarization.
    document (str): the path to the document. If provided, the text will be extracted from the document.

    """
    # Extract the text from the document, if provided
    if document:
        text = extract_text(document)

    # Initialize the tokenizer and the model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Initialize the summarizer
    summarizer = GPT3Summarizer(
        tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Generate the summary
    summary = summarizer(text, max_length=max_length, num_beams=2, min_length=30)

    # Generate the text using the local Llama model
    data = {"text": f"{text}\n\nSummary:\n\n{summary}", "top_k": 1}
    headers = {"Content-Type": "application/json"}
    response = requests.post(
        "http://localhost:11434/generate", data=json.dumps(data), headers=headers
    )
    generated_text = response.json()["text"]

    return generated_text
