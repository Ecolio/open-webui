import logging
from typing import List
from utils.utils import (
    query_doc,
    query_doc_with_hybrid_search,
    query_collection,
    query_collection_with_hybrid_search,
)
from pydantic import BaseModel
from fastapi import HTTPException
from constants import ERROR_MESSAGES

log = logging.getLogger(__name__)


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


class QueryCollectionsForm(BaseModel):
    collection_names: List[str]
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


def query_doc_handler(form_data: QueryDocForm, app_state):
    try:
        if app_state.ENABLE_RAG_HYBRID_SEARCH:
            return query_doc_with_hybrid_search(
                collection_name=form_data.collection_name,
                query=form_data.query,
                embedding_function=app_state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app_state.TOP_K,
                reranking_function=app_state.sentence_transformer_rf,
                r=form_data.r if form_data.r else app_state.RELEVANCE_THRESHOLD,
            )
        else:
            return query_doc(
                collection_name=form_data.collection_name,
                query=form_data.query,
                embedding_function=app_state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app_state.TOP_K,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


def query_collection_handler(form_data: QueryCollectionsForm, app_state):
    try:
        if app_state.ENABLE_RAG_HYBRID_SEARCH:
            return query_collection_with_hybrid_search(
                collection_names=form_data.collection_names,
                query=form_data.query,
                embedding_function=app_state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app_state.TOP_K,
                reranking_function=app_state.sentence_transformer_rf,
                r=form_data.r if form_data.r else app_state.RELEVANCE_THRESHOLD,
            )
        else:
            return query_collection(
                collection_names=form_data.collection_names,
                query=form_data.query,
                embedding_function=app_state.EMBEDDING_FUNCTION,
                k=form_data.k if form_data.k else app_state.TOP_K,
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )
