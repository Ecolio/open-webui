import logging
from langchain_community.document_loaders import UnstructuredMarkdownLoader

log = logging.getLogger(__name__)


def handle_markdown(file_path):
    return UnstructuredMarkdownLoader(file_path)
