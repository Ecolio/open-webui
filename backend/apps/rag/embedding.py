import logging
from apps.ollama.main import generate_ollama_embeddings, GenerateEmbeddingsForm
from utils.misc import generate_openai_embeddings

log = logging.getLogger(__name__)


def get_embedding_function(
    embedding_engine,
    embedding_model,
    embedding_function,
    openai_key,
    openai_url,
):
    def generate_multiple(query, f):
        if isinstance(query, list):
            return [f(q) for q in query]
        else:
            return f(query)

    def ollama_func(query):
        return generate_ollama_embeddings(
            GenerateEmbeddingsForm(
                model=embedding_model,
                prompt=query,
            )
        )

    def openai_func(query):
        return generate_openai_embeddings(
            model=embedding_model,
            text=query,
            key=openai_key,
            url=openai_url,
        )

    if embedding_engine == "":

        def func(query):
            return embedding_function.encode(query).tolist()

        return func
    elif embedding_engine == "ollama":
        return lambda query: generate_multiple(query, ollama_func)
    elif embedding_engine == "openai":
        return lambda query: generate_multiple(query, openai_func)

    raise ValueError(f"Unsupported embedding engine: {embedding_engine}")
