from abc import ABC, abstractmethod
from typing import List
from pydantic import BaseModel

class PaperMetadata(BaseModel):
    id: str
    title: str
    download_url: str
    doi: str

class PaperProvider(ABC):
    @abstractmethod
    def fetch_recent_papers(self, limit: int) -> List[PaperMetadata]:
        pass