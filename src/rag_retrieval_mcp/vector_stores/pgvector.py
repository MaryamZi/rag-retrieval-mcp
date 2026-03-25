import asyncio
from functools import partial

import psycopg2
import psycopg2.extras

from rag_retrieval_mcp.vector_stores.base import QueryResult, VectorStore


class PgVectorStore(VectorStore):
    def __init__(
        self,
        connection_string: str,
        table: str = "embeddings",
        text_column: str = "text",
        embedding_column: str = "embedding",
    ):
        self.connection_string = connection_string
        self.table = table
        self.text_column = text_column
        self.embedding_column = embedding_column

    def _query_sync(self, vector: list[float], top_k: int) -> list[QueryResult]:
        conn = psycopg2.connect(self.connection_string)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT *, 1 - ({self.embedding_column} <=> %s::vector) AS score "
                    f"FROM {self.table} "
                    f"ORDER BY {self.embedding_column} <=> %s::vector "
                    f"LIMIT %s",
                    (str(vector), str(vector), top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        exclude = {self.text_column, self.embedding_column, "score", "metadata"}

        results = []
        for row in rows:
            text = row[self.text_column]
            if not text:
                continue

            metadata = dict(row.get("metadata") or {}) if "metadata" in row else {}
            for key, val in row.items():
                if key not in exclude:
                    metadata[key] = val

            results.append(QueryResult(text=text, score=row["score"], metadata=metadata))
        return results

    async def query(self, vector: list[float], top_k: int = 5) -> list[QueryResult]:
        return await asyncio.to_thread(partial(self._query_sync, vector, top_k))
