import asyncio
from pathlib import Path

from src.services.chemrxiv_client import ChemRxivClient
from src.services.downloader import PDFDownloader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Pipeline:

    def __init__(self):

        self.client = ChemRxivClient()
        self.downloader = PDFDownloader()

    async def run(self):

        papers = await self.client.get_last_month_papers()

        tasks = []

        for paper in papers:

            pdf_path = Path("data/raw_papers2") / f"{paper['id']}.pdf"

            tasks.append(
                self.downloader.download_pdf(
                    paper["pdf_url"],
                    pdf_path
                )
            )

        await asyncio.gather(*tasks)

        logger.info("Download pipeline complete")