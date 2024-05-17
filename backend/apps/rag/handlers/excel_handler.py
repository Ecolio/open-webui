import logging
from langchain_community.document_loaders import UnstructuredExcelLoader

log = logging.getLogger(__name__)


def handle_excel(file_path):
    return UnstructuredExcelLoader(file_path)
