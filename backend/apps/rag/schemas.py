from pydantic import BaseModel
from typing import Optional, List


class CollectionNameForm(BaseModel):
    collection_name: Optional[str] = "test"


class UrlForm(CollectionNameForm):
    url: str


class OpenAIConfigForm(BaseModel):
    url: str
    key: str


class EmbeddingModelUpdateForm(BaseModel):
    openai_config: Optional[OpenAIConfigForm] = None
    embedding_engine: str
    embedding_model: str


class RerankingModelUpdateForm(BaseModel):
    reranking_model: str


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


class QuerySettingsForm(BaseModel):
    k: Optional[int] = None
    r: Optional[float] = None
    template: Optional[str] = None
    hybrid: Optional[bool] = None


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


class TextRAGForm(BaseModel):
    name: str
    content: str
    collection_name: Optional[str] = None
