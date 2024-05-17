import logging
from langchain_community.document_loaders import UnstructuredXMLLoader

log = logging.getLogger(__name__)


def handle_xml(file_path):
    return UnstructuredXMLLoader(file_path)
