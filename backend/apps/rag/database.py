import logging
import sentence_transformers
from config import (
    DEVICE_TYPE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
)
from utils.utils import get_model_path

log = logging.getLogger(__name__)


def update_embedding_model(
    embedding_model: str,
    update_model: bool = False,
):
    if embedding_model:
        return sentence_transformers.SentenceTransformer(
            get_model_path(embedding_model, update_model),
            device=DEVICE_TYPE,
            trust_remote_code=RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
        )
    return None


def update_reranking_model(
    reranking_model: str,
    update_model: bool = False,
):
    if reranking_model:
        return sentence_transformers.CrossEncoder(
            get_model_path(reranking_model, update_model),
            device=DEVICE_TYPE,
            trust_remote_code=RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
        )
    return None
