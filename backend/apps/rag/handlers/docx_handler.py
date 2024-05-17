import logging
from langchain_community.document_loaders import Docx2txtLoader

log = logging.getLogger(__name__)


def store_doc_handler(file_path):
    return Docx2txtLoader(file_path)
