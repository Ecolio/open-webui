from fastapi import FastAPI
from apps.rag.main import (
    get_status,
    get_embedding_config,
    get_reraanking_config,
    update_embedding_config,
    update_reranking_config,
    get_rag_config,
    update_rag_config,
    get_rag_template,
    get_query_settings,
    update_query_settings,
    query_doc,
    query_collection,
    store_youtube_video,
    store_web,
    store_doc,
    store_text,
    scan_docs_dir,
    reset_vector_db,
    reset,
)

app = FastAPI()

app.add_api_route("/", get_status, methods=["GET"])
app.add_api_route("/embedding", get_embedding_config, methods=["GET"])
app.add_api_route("/reranking", get_reraanking_config, methods=["GET"])
app.add_api_route("/embedding/update", update_embedding_config, methods=["POST"])
app.add_api_route("/reranking/update", update_reranking_config, methods=["POST"])
app.add_api_route("/config", get_rag_config, methods=["GET"])
app.add_api_route("/config/update", update_rag_config, methods=["POST"])
app.add_api_route("/template", get_rag_template, methods=["GET"])
app.add_api_route("/query/settings", get_query_settings, methods=["GET"])
app.add_api_route("/query/settings/update", update_query_settings, methods=["POST"])
app.add_api_route("/query/doc", query_doc, methods=["POST"])
app.add_api_route("/query/collection", query_collection, methods=["POST"])
app.add_api_route("/youtube", store_youtube_video, methods=["POST"])
app.add_api_route("/web", store_web, methods=["POST"])
app.add_api_route("/doc", store_doc, methods=["POST"])
app.add_api_route("/text", store_text, methods=["POST"])
app.add_api_route("/scan", scan_docs_dir, methods=["GET"])
app.add_api_route("/reset/db", reset_vector_db, methods=["GET"])
app.add_api_route("/reset", reset, methods=["GET"])
