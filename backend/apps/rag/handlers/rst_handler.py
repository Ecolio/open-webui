import logging
from langchain_community.document_loaders import UnstructuredRSTLoader

log = logging.getLogger(__name__)


def handle_rst(file_path):
    return UnstructuredRSTLoader(file_path, mode="elements")
