import os

# Configuration settings for the RAG application

SRC_LOG_LEVELS = {
    "RAG": os.getenv("RAG_LOG_LEVEL", "INFO"),
}

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
DOCS_DIR = os.getenv("DOCS_DIR", "./data/docs")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", 0.5))
RAG_EMBEDDING_ENGINE = os.getenv("RAG_EMBEDDING_ENGINE", "")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_EMBEDDING_MODEL_AUTO_UPDATE = bool(
    os.getenv("RAG_EMBEDDING_MODEL_AUTO_UPDATE", True)
)
RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE = bool(
    os.getenv("RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE", False)
)
ENABLE_RAG_HYBRID_SEARCH = bool(os.getenv("ENABLE_RAG_HYBRID_SEARCH", True))
ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = bool(
    os.getenv("ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION", True)
)
RAG_RERANKING_MODEL = os.getenv(
    "RAG_RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
)
PDF_EXTRACT_IMAGES = bool(os.getenv("PDF_EXTRACT_IMAGES", False))
RAG_RERANKING_MODEL_AUTO_UPDATE = bool(
    os.getenv("RAG_RERANKING_MODEL_AUTO_UPDATE", True)
)
RAG_RERANKING_MODEL_TRUST_REMOTE_CODE = bool(
    os.getenv("RAG_RERANKING_MODEL_TRUST_REMOTE_CODE", False)
)
RAG_OPENAI_API_BASE_URL = os.getenv(
    "RAG_OPENAI_API_BASE_URL", "https://api.openai.com/v1"
)
RAG_OPENAI_API_KEY = os.getenv("RAG_OPENAI_API_KEY", "")
DEVICE_TYPE = os.getenv("DEVICE_TYPE", "cpu")
CHROMA_CLIENT = os.getenv("CHROMA_CLIENT", "chromadb.Client()")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
RAG_TEMPLATE = os.getenv("RAG_TEMPLATE", "[context]\n\n[query]")
ENABLE_RAG_LOCAL_WEB_FETCH = bool(os.getenv("ENABLE_RAG_LOCAL_WEB_FETCH", False))
YOUTUBE_LOADER_LANGUAGE = os.getenv("YOUTUBE_LOADER_LANGUAGE", "en")
