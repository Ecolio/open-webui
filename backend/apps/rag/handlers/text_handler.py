import logging
from langchain_community.document_loaders import TextLoader

log = logging.getLogger(__name__)


def handle_text(file_path):
    return TextLoader(file_path, autodetect_encoding=True)
