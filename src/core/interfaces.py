from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from pathlib import Path

from .models import PaperMetadata


class PaperSource(ABC):
    """
    Abstract interface for paper sources.

    Strategy Pattern:
    Allows interchangeable implementations (ChemRxiv, arXiv, PubMed, etc.)
    """

    @abstractmethod
    def fetch_papers(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> List[PaperMetadata]:
        """
        Fetch papers published between given dates.

        Returns:
            List of PaperMetadata
        """
        pass


class PaperDownloader(ABC):
    """
    Interface for downloading paper PDFs.
    """

    @abstractmethod
    def download(self, paper: PaperMetadata) -> Path:
        """
        Download a paper PDF.

        Returns:
            Path to downloaded file
        """
        pass


class StorageManager(ABC):
    """
    Interface for managing storage.
    """

    @abstractmethod
    def get_pdf_path(self, paper: PaperMetadata) -> Path:
        """
        Return path where paper should be stored.
        """
        pass

    @abstractmethod
    def exists(self, paper: PaperMetadata) -> bool:
        """
        Check if paper already exists.
        """
        pass