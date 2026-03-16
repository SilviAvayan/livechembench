"""Core interfaces for paper acquisition providers."""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel


class PaperMetadata(BaseModel):
    """Normalized metadata for a single paper."""

    id: str
    title: str
    download_url: str
    doi: str


class PaperProvider(ABC):
    """Protocol for any source of scientific papers (e.g., ChemRxiv, PubMed/PMC)."""

    @abstractmethod
    def fetch_recent_papers(self, limit: int) -> List[PaperMetadata]:
        """Return up to `limit` recent papers that match the global search criteria."""
        raise NotImplementedError