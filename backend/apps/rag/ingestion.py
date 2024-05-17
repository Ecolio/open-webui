import logging
import os
import mimetypes
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException, status, File, Form
from config import UPLOAD_DIR, DOCS_DIR, CHROMA_CLIENT, ERROR_MESSAGES
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.misc import (
    calculate_sha256,
    calculate_sha256_string,
    sanitize_filename,
    extract_folders_after_data_docs,
)
from apps.web.models.documents import Documents, DocumentForm
from database import update_embedding_model, update_reranking_model
from embedding import get_embedding_function
from apps.rag.main import app  # Import the app instance

from handlers.pdf_handler import handle_pdf
from handlers.csv_handler import handle_csv
from handlers.web_handler import handle_html
from handlers.youtube_handler import store_youtube_video_handler
from handlers.docx_handler import store_doc_handler
from handlers.epub_handler import handle_epub
from handlers.excel_handler import handle_excel
from handlers.markdown_handler import handle_markdown
from handlers.rst_handler import handle_rst
from handlers.text_handler import handle_text
from handlers.xml_handler import handle_xml
from handlers.source_code_handler import handle_source_code
from utils import SentenceTextSplitter  # Import SentenceTextSplitter
from langchain_core.documents import Document  # Import Document
from chromadb.utils.batch_utils import create_batches

log = logging.getLogger(__name__)


def store_data_in_vector_db(data, collection_name, overwrite: bool = False) -> bool:
    text_splitter = SentenceTextSplitter(
        sentences_per_chunk=5,
        chunk_overlap=1,
    )

    docs = []
    for document in data:
        chunks = text_splitter.split_text(document.page_content)
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata=document.metadata))

    if len(docs) > 0:
        log.info(f"store_data_in_vector_db {docs}")
        return store_docs_in_vector_db(docs, collection_name, overwrite), None
    else:
        raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)


def store_text_in_vector_db(
    text, metadata, collection_name, overwrite: bool = False
) -> bool:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5,
        chunk_overlap=1,
        add_start_index=True,
    )
    docs = text_splitter.create_documents([text], metadatas=[metadata])
    return store_docs_in_vector_db(docs, collection_name, overwrite)


def store_docs_in_vector_db(docs, collection_name, overwrite: bool = False) -> bool:
    log.info(f"store_docs_in_vector_db {docs} {collection_name}")

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    try:
        if overwrite:
            for collection in CHROMA_CLIENT.list_collections():
                if collection_name == collection.name:
                    log.info(f"deleting existing collection {collection_name}")
                    CHROMA_CLIENT.delete_collection(name=collection_name)

        collection = CHROMA_CLIENT.create_collection(name=collection_name)

        embedding_func = get_embedding_function(
            app.state.RAG_EMBEDDING_ENGINE,
            app.state.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            app.state.OPENAI_API_KEY,
            app.state.OPENAI_API_BASE_URL,
        )

        embedding_texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = embedding_func(embedding_texts)

        for batch in create_batches(
            api=CHROMA_CLIENT,
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=metadatas,
            embeddings=embeddings,
            documents=texts,
        ):
            collection.add(*batch)

        return True
    except Exception as e:
        log.exception(e)
        if e.__class__.__name__ == "UniqueConstraintError":
            return True

        return False


def get_loader(filename: str, file_content_type: str, file_path: str):
    file_ext = filename.split(".")[-1].lower()

    if file_ext == "pdf":
        return handle_pdf(file_path), True
    elif file_ext == "csv":
        return handle_csv(file_path), True
    elif file_ext == "rst":
        return handle_rst(file_path), True
    elif file_ext == "xml":
        return handle_xml(file_path), True
    elif file_ext in ["htm", "html"]:
        return handle_html(file_path), True
    elif file_ext == "md":
        return handle_markdown(file_path), True
    elif file_content_type == "application/epub+zip":
        return handle_epub(file_path), True
    elif (
        file_content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or file_ext in ["doc", "docx"]
    ):
        return store_doc_handler(file_path), True
    elif file_content_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ] or file_ext in ["xls", "xlsx"]:
        return handle_excel(file_path), True
    elif file_ext in handle_source_code or (
        file_content_type and file_content_type.find("text/") >= 0
    ):
        return handle_source_code(file_path), True
    else:
        return handle_text(file_path), False
