import logging
from langchain_community.document_loaders import PyPDFLoader
from apps.rag.main import app

log = logging.getLogger(__name__)


def handle_pdf(file_path):
    return PyPDFLoader(file_path, extract_images=app.state.PDF_EXTRACT_IMAGES)
