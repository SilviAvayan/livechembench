from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass(frozen=True)
class PaperMetadata:
    """
    Represents metadata for a scientific paper.

    This is the core domain object used throughout the pipeline.
    """

    id: str
    title: str
    authors: List[str]
    published_date: datetime
    pdf_url: str
    source: str

    def filename(self) -> str:
        """
        Generate a safe filename for the PDF.

        Returns:
            str: filename like "chemrxiv_12345.pdf"
        """
        return f"{self.source}_{self.id}.pdf"