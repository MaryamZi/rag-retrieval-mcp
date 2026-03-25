"""Tests for the RAG retrieval MCP server."""

import json
import os
from unittest.mock import patch

import pytest

from rag_retrieval_mcp.embedding_providers.base import EmbeddingProvider
from rag_retrieval_mcp.vector_stores.base import QueryResult, VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStore):
    def __init__(self, results: list[QueryResult] | None = None):
        self.results = results or []
        self.last_query_vector: list[float] | None = None
        self.last_top_k: int | None = None

    async def query(self, vector: list[float], top_k: int = 5) -> list[QueryResult]:
        self.last_query_vector = vector
        self.last_top_k = top_k
        return self.results


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset cached provider/store between tests."""
    import rag_retrieval_mcp.server as srv

    srv._embedding_provider = None
    srv._vector_store = None
    yield
    srv._embedding_provider = None
    srv._vector_store = None


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_returns_structured_json(self):
        from rag_retrieval_mcp.server import retrieve

        results = [
            QueryResult(text="first chunk", score=0.95, metadata={"source": "doc1"}),
            QueryResult(text="second chunk", score=0.80, metadata={"source": "doc2"}),
        ]
        fake_store = FakeVectorStore(results)
        fake_provider = FakeEmbeddingProvider()

        import rag_retrieval_mcp.server as srv

        srv._embedding_provider = fake_provider
        srv._vector_store = fake_store

        output = await retrieve("test query")
        parsed = json.loads(output)

        assert len(parsed) == 2
        assert parsed[0]["text"] == "first chunk"
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["metadata"] == {"source": "doc1"}
        assert parsed[1]["text"] == "second chunk"

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self):
        from rag_retrieval_mcp.server import retrieve

        import rag_retrieval_mcp.server as srv

        srv._embedding_provider = FakeEmbeddingProvider()
        srv._vector_store = FakeVectorStore([])

        output = await retrieve("test query")
        assert output == "No relevant content found."

    @pytest.mark.asyncio
    async def test_retrieve_passes_vector_and_top_k(self):
        from rag_retrieval_mcp.server import retrieve

        fake_store = FakeVectorStore([])
        fake_provider = FakeEmbeddingProvider()

        import rag_retrieval_mcp.server as srv

        srv._embedding_provider = fake_provider
        srv._vector_store = fake_store

        with patch.dict(os.environ, {"RAG_TOP_K": "10"}):
            await retrieve("hello")

        assert fake_store.last_query_vector == [0.1, 0.2, 0.3]
        assert fake_store.last_top_k == 10


class TestConfig:
    def test_invalid_top_k_raises(self):
        from rag_retrieval_mcp.server import _get_top_k

        with patch.dict(os.environ, {"RAG_TOP_K": "abc"}):
            with pytest.raises(ValueError, match="must be an integer"):
                _get_top_k()

    def test_negative_top_k_raises(self):
        from rag_retrieval_mcp.server import _get_top_k

        with patch.dict(os.environ, {"RAG_TOP_K": "0"}):
            with pytest.raises(ValueError, match="must be >= 1"):
                _get_top_k()

    def test_unknown_embedding_provider_raises(self):
        from rag_retrieval_mcp.server import _get_embedding_provider

        with patch.dict(os.environ, {"RAG_EMBEDDING_PROVIDER": "unknown"}):
            with pytest.raises(ValueError, match="Unknown embedding provider"):
                _get_embedding_provider()

    def test_unknown_vector_store_raises(self):
        from rag_retrieval_mcp.server import _get_vector_store

        with patch.dict(os.environ, {"RAG_VECTOR_STORE": "unknown"}):
            with pytest.raises(ValueError, match="Unknown vector store"):
                _get_vector_store()

    def test_missing_openai_api_key_raises(self):
        from rag_retrieval_mcp.server import _get_embedding_provider

        env = {"RAG_EMBEDDING_PROVIDER": "openai"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                _get_embedding_provider()

    def test_missing_pinecone_api_key_raises(self):
        from rag_retrieval_mcp.server import _get_vector_store

        env = {"RAG_VECTOR_STORE": "pinecone"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
                _get_vector_store()

    def test_missing_pgvector_connection_string_raises(self):
        from rag_retrieval_mcp.server import _get_vector_store

        env = {"RAG_VECTOR_STORE": "pgvector"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="PGVECTOR_CONNECTION_STRING"):
                _get_vector_store()


class TestEmbeddingProviderBatch:
    @pytest.mark.asyncio
    async def test_default_batch_calls_embed(self):
        provider = FakeEmbeddingProvider()
        results = await provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r == [0.1, 0.2, 0.3] for r in results)
