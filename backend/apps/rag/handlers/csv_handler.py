import logging
from langchain_community.document_loaders import CSVLoader

log = logging.getLogger(__name__)


def handle_csv(file_path):
    return CSVLoader(file_path)
