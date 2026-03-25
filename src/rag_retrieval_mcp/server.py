"""Generic RAG MCP server with pluggable embedding providers and vector stores."""

import json
import os

from mcp.server.fastmcp import FastMCP

from rag_retrieval_mcp.embedding_providers.base import EmbeddingProvider
from rag_retrieval_mcp.vector_stores.base import VectorStore

mcp = FastMCP("rag")

_embedding_provider: EmbeddingProvider | None = None
_vector_store: VectorStore | None = None


def _get_top_k() -> int:
    raw = os.environ.get("RAG_TOP_K", "5")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"RAG_TOP_K must be an integer, got: {raw!r}")
    if value < 1:
        raise ValueError(f"RAG_TOP_K must be >= 1, got: {value}")
    return value


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_embedding_provider() -> EmbeddingProvider:
    provider = os.environ.get("RAG_EMBEDDING_PROVIDER", "openai")
    if provider == "openai":
        api_key = _require_env("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        from rag_retrieval_mcp.embedding_providers.openai import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(api_key=api_key, model=model)
    raise ValueError(
        f"Unknown embedding provider: {provider!r}. Supported: 'openai'"
    )


def _get_vector_store() -> VectorStore:
    store = os.environ.get("RAG_VECTOR_STORE", "pinecone")
    if store == "pinecone":
        api_key = _require_env("PINECONE_API_KEY")
        host = _require_env("PINECONE_HOST")
        text_field = os.environ.get("PINECONE_TEXT_FIELD", "text")
        from rag_retrieval_mcp.vector_stores.pinecone import PineconeVectorStore

        return PineconeVectorStore(api_key=api_key, host=host, text_field=text_field)
    if store == "pgvector":
        connection_string = _require_env("PGVECTOR_CONNECTION_STRING")
        table = os.environ.get("PGVECTOR_TABLE", "embeddings")
        text_column = os.environ.get("PGVECTOR_TEXT_COLUMN", "text")
        embedding_column = os.environ.get("PGVECTOR_EMBEDDING_COLUMN", "embedding")
        from rag_retrieval_mcp.vector_stores.pgvector import PgVectorStore

        return PgVectorStore(
            connection_string=connection_string,
            table=table,
            text_column=text_column,
            embedding_column=embedding_column,
        )
    raise ValueError(
        f"Unknown vector store: {store!r}. Supported: 'pinecone', 'pgvector'"
    )


def get_embedding_provider() -> EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = _get_embedding_provider()
    return _embedding_provider


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = _get_vector_store()
    return _vector_store


@mcp.tool()
async def retrieve(query: str) -> str:
    """Search a knowledge base and return relevant content.

    Args:
        query: The search query to find relevant content.
    """
    top_k = _get_top_k()
    provider = get_embedding_provider()
    store = get_vector_store()

    vector = await provider.embed(query)
    results = await store.query(vector, top_k=top_k)

    if not results:
        return "No relevant content found."

    return json.dumps(
        [
            {"text": r.text, "score": r.score, "metadata": r.metadata}
            for r in results
        ],
        indent=2,
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
