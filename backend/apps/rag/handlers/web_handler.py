import logging
import validators
import urllib.parse
import socket
from langchain_community.document_loaders import BSHTMLLoader, WebBaseLoader
from constants import ERROR_MESSAGES
from config import ENABLE_RAG_LOCAL_WEB_FETCH

log = logging.getLogger(__name__)


def store_web_handler(url, verify_ssl=True):
    if isinstance(validators.url(url), validators.ValidationError):
        raise ValueError(ERROR_MESSAGES.INVALID_URL)
    if not ENABLE_RAG_LOCAL_WEB_FETCH:
        parsed_url = urllib.parse.urlparse(url)
        ipv4_addresses, ipv6_addresses = resolve_hostname(parsed_url.hostname)
        for ip in ipv4_addresses:
            if validators.ipv4(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
        for ip in ipv6_addresses:
            if validators.ipv6(ip, private=True):
                raise ValueError(ERROR_MESSAGES.INVALID_URL)
    return WebBaseLoader(url, verify_ssl=verify_ssl)


def resolve_hostname(hostname):
    addr_info = socket.getaddrinfo(hostname, None)
    ipv4_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET]
    ipv6_addresses = [info[4][0] for info in addr_info if info[0] == socket.AF_INET6]
    return ipv4_addresses, ipv6_addresses


def handle_html(file_path):
    return BSHTMLLoader(file_path, open_encoding="unicode_escape")
