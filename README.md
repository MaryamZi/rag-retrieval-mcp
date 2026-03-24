# MCP Server for RAG Retrieval

A generic Retrieval-Augmented Generation (RAG) Model Context Protocol (MCP) server with pluggable embedding providers and vector stores.

## Why this server?

Vendor MCP servers usually only support their (own) integrated embedding models. If your index uses external embeddings (e.g., OpenAI), those servers can't query it. This server fills that gap — it embeds your query with the provider of your choice, then searches any supported vector store.

## Currently Supports

**Embedding Providers:**
- OpenAI (`text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`, etc.)

**Vector Stores:**
- Pinecone

## Tools

### `retrieve`

Search a knowledge base and return relevant content.

**Parameters:**
- `query` (string, required) — The search query to find relevant content.

**Returns** a JSON array of results, each with `text`, `score`, and `metadata` fields.

## Install & Run

Run directly with `uvx` (no install needed):

```bash
uvx rag-retrieval-mcp[all]
```

Or install with pip:

```bash
pip install rag-retrieval-mcp[all]
rag-retrieval-mcp
```

### MCP client configuration

```json
{
  "mcpServers": {
    "rag-retrieval": {
      "command": "uvx",
      "args": ["rag-retrieval-mcp[all]"],
      "env": {
        "OPENAI_API_KEY": "your-openai-api-key",
        "PINECONE_API_KEY": "your-pinecone-api-key",
        "PINECONE_HOST": "your-pinecone-index-host-url"
      }
    }
  }
}
```

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RAG_EMBEDDING_PROVIDER` | No | `openai` | Embedding provider to use |
| `RAG_VECTOR_STORE` | No | `pinecone` | Vector store to use |
| `RAG_TOP_K` | No | `5` | Number of results to return |
| `OPENAI_API_KEY` | Yes (if using OpenAI) | | OpenAI API key |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | OpenAI embedding model |
| `PINECONE_API_KEY` | Yes (if using Pinecone) | | Pinecone API key |
| `PINECONE_HOST` | Yes (if using Pinecone) | | Pinecone index host URL |
| `PINECONE_TEXT_FIELD` | No | `text` | Metadata field containing text |

## Adding New Providers

Implement the `EmbeddingProvider` or `VectorStore` abstract base class and register it in `server.py`'s factory function. See `src/rag_retrieval_mcp/embedding_providers/base.py` and `src/rag_retrieval_mcp/vector_stores/base.py` for the interfaces.

## License

Apache License 2.0
