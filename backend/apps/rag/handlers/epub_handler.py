import logging
from langchain_community.document_loaders import Docx2txtLoader

log = logging.getLogger(__name__)


def handle_docx(file_path):
    return Docx2txtLoader(file_path)
