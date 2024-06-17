import os
import logging
import requests
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from typing import List, Optional
from config import SRC_LOG_LEVELS, CHROMA_CLIENT
from sentence_transformers import util
import operator
from typing import Any, Sequence
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.pydantic_v1 import Extra
import nltk
from nltk.tokenize import sent_tokenize
from constants import ERROR_MESSAGES




log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])


def query_doc(collection_name: str, query: str, embedding_function, k: int):
    try:
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        query_embeddings = embedding_function(query)

        result = collection.query(
            query_embeddings=[query_embeddings],
            n_results=k,
        )

        log.info(f"query_doc:result {result}")
        return result
    except Exception as e:
        raise e


def query_doc_with_hybrid_search(
    collection_name: str,
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    try:
        collection = CHROMA_CLIENT.get_collection(name=collection_name)
        documents = collection.get()

        bm25_retriever = BM25Retriever.from_texts(
            texts=documents.get("documents"),
            metadatas=documents.get("metadatas"),
        )
        bm25_retriever.k = k

        chroma_retriever = ChromaRetriever(
            collection=collection,
            embedding_function=embedding_function,
            top_n=k,
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
        )

        compressor = RerankCompressor(
            embedding_function=embedding_function,
            top_n=k,
            reranking_function=reranking_function,
            r_score=r,
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )

        result = compression_retriever.invoke(query)
        result = {
            "distances": [[d.metadata.get("score") for d in result]],
            "documents": [[d.page_content for d in result]],
            "metadatas": [[d.metadata for d in result]],
        }

        log.info(f"query_doc_with_hybrid_search:result {result}")
        return result
    except Exception as e:
        raise e


def query_collection(
    collection_names: List[str], query: str, embedding_function, k: int
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc(
                collection_name=collection_name,
                query=query,
                k=k,
                embedding_function=embedding_function,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k)


def query_collection_with_hybrid_search(
    collection_names: List[str],
    query: str,
    embedding_function,
    k: int,
    reranking_function,
    r: float,
):
    results = []
    for collection_name in collection_names:
        try:
            result = query_doc_with_hybrid_search(
                collection_name=collection_name,
                query=query,
                embedding_function=embedding_function,
                k=k,
                reranking_function=reranking_function,
                r=r,
            )
            results.append(result)
        except:
            pass
    return merge_and_sort_query_results(results, k=k, reverse=True)


def merge_and_sort_query_results(query_results, k, reverse=False):
    combined_distances = []
    combined_documents = []
    combined_metadatas = []

    for data in query_results:
        combined_distances.extend(data["distances"][0])
        combined_documents.extend(data["documents"][0])
        combined_metadatas.extend(data["metadatas"][0])

    combined = list(zip(combined_distances, combined_documents, combined_metadatas))

    combined.sort(key=lambda x: x[0], reverse=reverse)

    if not combined:
        sorted_distances = []
        sorted_documents = []
        sorted_metadatas = []
    else:
        sorted_distances, sorted_documents, sorted_metadatas = zip(*combined)
        sorted_distances = list(sorted_distances)[:k]
        sorted_documents = list(sorted_documents)[:k]
        sorted_metadatas = list(sorted_metadatas)[:k]

    result = {
        "distances": [sorted_distances],
        "documents": [sorted_documents],
        "metadatas": [sorted_metadatas],
    }

    return result


def get_model_path(model: str, update_model: bool = False):
    cache_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME")

    local_files_only = not update_model

    snapshot_kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }

    log.debug(f"model: {model}")
    log.debug(f"snapshot_kwargs: {snapshot_kwargs}")

    if (
        os.path.exists(model)
        or ("\\" in model or model.count("/") > 1)
        and local_files_only
    ):
        return model
    elif "/" not in model:
        model = "sentence-transformers" + "/" + model

    snapshot_kwargs["repo_id"] = model

    try:
        model_repo_path = snapshot_download(**snapshot_kwargs)
        log.debug(f"model_repo_path: {model_repo_path}")
        return model_repo_path
    except Exception as e:
        log.exception(f"Cannot determine model snapshot path: {e}")
        return model


def generate_openai_embeddings(
    model: str, text: str, key: str, url: str = "https://api.openai.com/v1"
):
    try:
        r = requests.post(
            f"{url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json={"input": text, "model": model},
        )
        r.raise_for_status()
        data = r.json()
        if "data" in data:
            return data["data"][0]["embedding"]
        else:
            raise "Something went wrong :/"
    except Exception as e:
        print(e)
        return None


class ChromaRetriever(BaseRetriever):
    collection: Any
    embedding_function: Any
    top_n: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        query_embeddings = self.embedding_function(query)

        results = self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=self.top_n,
        )

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        results = []
        for idx in range(len(ids)):
            results.append(
                Document(
                    metadata=metadatas[idx],
                    page_content=documents[idx],
                )
            )
        return results


class RerankCompressor(BaseDocumentCompressor):
    embedding_function: Any
    top_n: int
    reranking_function: Any
    r_score: float

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        reranking = self.reranking_function is not None

        if reranking:
            scores = self.reranking_function.predict(
                [(query, doc.page_content) for doc in documents]
            )
        else:
            query_embedding = self.embedding_function(query)
            document_embedding = self.embedding_function(
                [doc.page_content for doc in documents]
            )
            scores = util.cos_sim(query_embedding, document_embedding)[0]

        docs_with_scores = list(zip(documents, scores.tolist()))
        if self.r_score:
            docs_with_scores = [
                (d, s) for d, s in docs_with_scores if s >= self.r_score
            ]

        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        final_results = []
        for doc, doc_score in result[: self.top_n]:
            metadata = doc.metadata
            metadata["score"] = doc_score
            doc = Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
            final_results.append(doc)
        return final_results


nltk.download("punkt")


class SentenceTextSplitter:
    def __init__(self, sentences_per_chunk: int, chunk_overlap: int):
        self.sentences_per_chunk = sentences_per_chunk
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str):
        sentences = sent_tokenize(text)
        chunks = []

        for i in range(
            0, len(sentences), self.sentences_per_chunk - self.chunk_overlap
        ):
            chunk = " ".join(sentences[i : i + self.sentences_per_chunk])
            chunks.append(chunk)

        return chunks
