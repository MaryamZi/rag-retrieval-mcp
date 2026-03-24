import asyncio
from functools import partial

from pinecone import Pinecone

from rag_retrieval_mcp.vector_stores.base import QueryResult, VectorStore


class PineconeVectorStore(VectorStore):
    def __init__(self, api_key: str, host: str, text_field: str = "text"):
        pc = Pinecone(api_key=api_key)
        self.index = pc.Index(host=host)
        self.text_field = text_field

    async def query(self, vector: list[float], top_k: int = 5) -> list[QueryResult]:
        results = await asyncio.to_thread(
            partial(
                self.index.query,
                vector=vector,
                top_k=top_k,
                include_metadata=True,
            )
        )
        return [
            QueryResult(
                text=match.metadata.get(self.text_field, ""),
                score=match.score,
                metadata={k: v for k, v in match.metadata.items() if k != self.text_field},
            )
            for match in results.matches
            if match.metadata.get(self.text_field)
        ]
