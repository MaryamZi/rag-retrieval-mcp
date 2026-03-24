from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    async def query(self, vector: list[float], top_k: int = 5) -> list[QueryResult]:
        """Query the vector store with an embedding vector and return matching results."""
        ...
