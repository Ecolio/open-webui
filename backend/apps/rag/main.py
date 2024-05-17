from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import shutil
from typing import List, Optional
from pydantic import BaseModel
from config import (
    SRC_LOG_LEVELS,
    UPLOAD_DIR,
    DOCS_DIR,
    RAG_TOP_K,
    RAG_RELEVANCE_THRESHOLD,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    RAG_RERANKING_MODEL,
    PDF_EXTRACT_IMAGES,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    DEVICE_TYPE,
    CHROMA_CLIENT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TEMPLATE,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    YOUTUBE_LOADER_LANGUAGE,
)
from constants import ERROR_MESSAGES
from apps.web.models.documents import Documents, DocumentForm
from utils.misc import (
    calculate_sha256,
    calculate_sha256_string,
    sanitize_filename,
    extract_folders_after_data_docs,
)
from utils.utils import get_current_user, get_admin_user
from ingestion import store_data_in_vector_db, store_text_in_vector_db, get_loader
from database import update_embedding_model, update_reranking_model
from embedding import get_embedding_function
from similarity import query_doc_handler, query_collection_handler
from handlers.youtube_handler import store_youtube_video_handler
from handlers.web_handler import store_web_handler
from handlers.docx_handler import store_doc_handler
from handlers.text_handler import store_text_handler
from handlers.scan_handler import scan_docs_dir_handler

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

app = FastAPI()

app.state.TOP_K = RAG_TOP_K
app.state.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
)
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.RAG_TEMPLATE = RAG_TEMPLATE
app.state.OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.OPENAI_API_KEY = RAG_OPENAI_API_KEY
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES
app.state.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.YOUTUBE_LOADER_TRANSLATION = None
app.state.SENTENCES_PER_CHUNK = 5

update_embedding_model(app.state.RAG_EMBEDDING_MODEL, RAG_EMBEDDING_MODEL_AUTO_UPDATE)
update_reranking_model(app.state.RAG_RERANKING_MODEL, RAG_RERANKING_MODEL_AUTO_UPDATE)
app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.RAG_EMBEDDING_ENGINE,
    app.state.RAG_EMBEDDING_MODEL,
    app.state.sentence_transformer_ef,
    app.state.OPENAI_API_KEY,
    app.state.OPENAI_API_BASE_URL,
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = "test"


class UrlForm(CollectionNameForm):
    url: str


@app.get("/")
async def get_status():
    return {
        "status": True,
        "chunk_size": app.state.CHUNK_SIZE,
        "chunk_overlap": app.state.CHUNK_OVERLAP,
        "template": app.state.RAG_TEMPLATE,
        "embedding_engine": app.state.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.RAG_EMBEDDING_MODEL,
        "reranking_model": app.state.RAG_RERANKING_MODEL,
    }


@app.get("/embedding")
async def get_embedding_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "embedding_engine": app.state.RAG_EMBEDDING_ENGINE,
        "embedding_model": app.state.RAG_EMBEDDING_MODEL,
        "openai_config": {
            "url": app.state.OPENAI_API_BASE_URL,
            "key": app.state.OPENAI_API_KEY,
        },
    }


@app.get("/reranking")
async def get_reraanking_config(user=Depends(get_admin_user)):
    return {"status": True, "reranking_model": app.state.RAG_RERANKING_MODEL}


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    embedding_engine: str
    embedding_model: str


@app.post("/embedding/update")
async def update_embedding_config(
    form_data: EmbeddingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating embedding model: {app.state.RAG_EMBEDDING_MODEL} to {form_data.embedding_model}"
    )
    try:
        app.state.RAG_EMBEDDING_ENGINE = form_data.embedding_engine
        app.state.RAG_EMBEDDING_MODEL = form_data.embedding_model

        if app.state.RAG_EMBEDDING_ENGINE in ["ollama", "openai"]:
            if form_data.openai_config != None:
                app.state.OPENAI_API_BASE_URL = form_data.openai_config.url
                app.state.OPENAI_API_KEY = form_data.openai_config.key

        update_embedding_model(app.state.RAG_EMBEDDING_MODEL, True)

        app.state.EMBEDDING_FUNCTION = get_embedding_function(
            app.state.RAG_EMBEDDING_ENGINE,
            app.state.RAG_EMBEDDING_MODEL,
            app.state.sentence_transformer_ef,
            app.state.OPENAI_API_KEY,
            app.state.OPENAI_API_BASE_URL,
        )

        return {
            "status": True,
            "embedding_engine": app.state.RAG_EMBEDDING_ENGINE,
            "embedding_model": app.state.RAG_EMBEDDING_MODEL,
            "openai_config": {
                "url": app.state.OPENAI_API_BASE_URL,
                "key": app.state.OPENAI_API_KEY,
            },
        }
    except Exception as e:
        log.exception(f"Problem updating embedding model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


class RerankingModelUpdateForm(BaseModel):
    reranking_model: str


@app.post("/reranking/update")
async def update_reranking_config(
    form_data: RerankingModelUpdateForm, user=Depends(get_admin_user)
):
    log.info(
        f"Updating reranking model: {app.state.RAG_RERANKING_MODEL} to {form_data.reranking_model}"
    )
    try:
        app.state.RAG_RERANKING_MODEL = form_data.reranking_model

        update_reranking_model(app.state.RAG_RERANKING_MODEL, True)

        return {
            "status": True,
            "reranking_model": app.state.RAG_RERANKING_MODEL,
        }
    except Exception as e:
        log.exception(f"Problem updating reranking model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@app.get("/config")
async def get_rag_config(user=Depends(get_admin_user)):
    return {
        "status": True,
        "pdf_extract_images": app.state.PDF_EXTRACT_IMAGES,
        "chunk": {
            "chunk_size": app.state.CHUNK_SIZE,
            "chunk_overlap": app.state.CHUNK_OVERLAP,
        },
        "web_loader_ssl_verification": app.state.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        "youtube": {
            "language": app.state.YOUTUBE_LOADER_LANGUAGE,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


class ChunkParamUpdateForm(BaseModel):
    chunk_size: int
    chunk_overlap: int


class YoutubeLoaderConfig(BaseModel):
    language: List[str]
    translation: Optional[str] = None


class ConfigUpdateForm(BaseModel):
    pdf_extract_images: Optional[bool] = None
    chunk: Optional[ChunkParamUpdateForm] = None
    web_loader_ssl_verification: Optional[bool] = None
    youtube: Optional[YoutubeLoaderConfig] = None


@app.post("/config/update")
async def update_rag_config(form_data: ConfigUpdateForm, user=Depends(get_admin_user)):
    app.state.PDF_EXTRACT_IMAGES = (
        form_data.pdf_extract_images
        if form_data.pdf_extract_images != None
        else app.state.PDF_EXTRACT_IMAGES
    )

    app.state.CHUNK_SIZE = (
        form_data.chunk.chunk_size if form_data.chunk != None else app.state.CHUNK_SIZE
    )

    app.state.CHUNK_OVERLAP = (
        form_data.chunk.chunk_overlap
        if form_data.chunk != None
        else app.state.CHUNK_OVERLAP
    )

    app.state.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
        form_data.web_loader_ssl_verification
        if form_data.web_loader_ssl_verification != None
        else app.state.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
    )

    app.state.YOUTUBE_LOADER_LANGUAGE = (
        form_data.youtube.language
        if form_data.youtube != None
        else app.state.YOUTUBE_LOADER_LANGUAGE
    )

    app.state.YOUTUBE_LOADER_TRANSLATION = (
        form_data.youtube.translation
        if form_data.youtube != None
        else app.state.YOUTUBE_LOADER_TRANSLATION
    )

    return {
        "status": True,
        "pdf_extract_images": app.state.PDF_EXTRACT_IMAGES,
        "chunk": {
            "chunk_size": app.state.CHUNK_SIZE,
            "chunk_overlap": app.state.CHUNK_OVERLAP,
        },
        "web_loader_ssl_verification": app.state.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
        "youtube": {
            "language": app.state.YOUTUBE_LOADER_LANGUAGE,
            "translation": app.state.YOUTUBE_LOADER_TRANSLATION,
        },
    }


@app.get("/template")
async def get_rag_template(user=Depends(get_current_user)):
    return {
        "status": True,
        "template": app.state.RAG_TEMPLATE,
    }


@app.get("/query/settings")
async def get_query_settings(user=Depends(get_admin_user)):
    return {
        "status": True,
        "template": app.state.RAG_TEMPLATE,
        "k": app.state.TOP_K,
        "r": app.state.RELEVANCE_THRESHOLD,
        "hybrid": app.state.ENABLE_RAG_HYBRID_SEARCH,
    }


class QuerySettingsForm(BaseModel):
    k: Optional[int] = None
    r: Optional[float] = None
    template: Optional[str] = None
    hybrid: Optional[bool] = None


@app.post("/query/settings/update")
async def update_query_settings(
    form_data: QuerySettingsForm, user=Depends(get_admin_user)
):
    app.state.RAG_TEMPLATE = form_data.template if form_data.template else RAG_TEMPLATE
    app.state.TOP_K = form_data.k if form_data.k else 4
    app.state.RELEVANCE_THRESHOLD = form_data.r if form_data.r else 0.0
    app.state.ENABLE_RAG_HYBRID_SEARCH = form_data.hybrid if form_data.hybrid else False
    return {
        "status": True,
        "template": app.state.RAG_TEMPLATE,
        "k": app.state.TOP_K,
        "r": app.state.RELEVANCE_THRESHOLD,
        "hybrid": app.state.ENABLE_RAG_HYBRID_SEARCH,
    }


class QueryDocForm(BaseModel):
    collection_name: str
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@app.post("/query/doc")
def query_doc(form_data: QueryDocForm, user=Depends(get_current_user)):
    return query_doc_handler(form_data, app.state)


class QueryCollectionsForm(BaseModel):
    collection_names: List[str]
    query: str
    k: Optional[int] = None
    r: Optional[float] = None
    hybrid: Optional[bool] = None


@app.post("/query/collection")
def query_collection(form_data: QueryCollectionsForm, user=Depends(get_current_user)):
    return query_collection_handler(form_data, app.state)


@app.post("/youtube")
def store_youtube_video(form_data: UrlForm, user=Depends(get_current_user)):
    return store_youtube_video_handler(form_data, app.state)


@app.post("/web")
def store_web(form_data: UrlForm, user=Depends(get_current_user)):
    return store_web_handler(form_data, app.state)


@app.post("/doc")
def store_doc(
    collection_name: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    return store_doc_handler(collection_name, file, user, app.state)


class TextRAGForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None


@app.post("/text")
def store_text(
    form_data: TextRAGForm,
    user=Depends(get_current_user),
):
    return store_text_handler(form_data, user, app.state)


@app.get("/scan")
def scan_docs_dir(user=Depends(get_admin_user)):
    return scan_docs_dir_handler(user, app.state)


@app.get("/reset/db")
def reset_vector_db(user=Depends(get_admin_user)):
    CHROMA_CLIENT.reset()


@app.get("/reset")
def reset(user=Depends(get_admin_user)) -> bool:
    folder = f"{UPLOAD_DIR}"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            log.error("Failed to delete %s. Reason: %s" % (file_path, e))

    try:
        CHROMA_CLIENT.reset()
    except Exception as e:
        log.exception(e)

    return True
