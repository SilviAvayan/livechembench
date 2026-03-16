from pathlib import Path
from typing import Dict

from src.config.loader import config
from src.core.interfaces import PaperProvider
from src.services.downloader import ProviderFactory
from src.utils.logger import logger


class DownloadPipeline:
    """
    Orchestrates end-to-end acquisition for all configured providers.

    Responsibilities:
    - build provider clients via `ProviderFactory`
    - request recent papers from each provider
    - persist the corresponding PDFs under `config.paths.raw_papers`
    """

    def __init__(self, provider_name: str | None = None) -> None:
        self.raw_path = Path(config.paths.raw_papers)
        # Ensure the target directory exists once on startup
        self.raw_path.mkdir(parents=True, exist_ok=True)

        # Registry of concrete provider clients
        if provider_name:
            key = provider_name.lower()
            self.providers: Dict[str, PaperProvider] = {
                key: ProviderFactory.get_provider(key),
            }
        else:
            self.providers = {
                "chemrxiv": ProviderFactory.get_provider("chemrxiv"),
                "pubmed": ProviderFactory.get_provider("pubmed"),
            }

    def download_pdf(self, url: str, filename: str) -> bool:
        """
        Download a single PDF to the pipeline's raw folder.

        This normalizes some URL schemes (e.g. ftp://) but otherwise delegates
        HTTP details to `requests`.
        """
        dest = self.raw_path / filename

        # Clean FTP or protocol-relative links
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("ftp://"):
            url = url.replace("ftp://", "https://")

        # If file already exists, avoid re-downloading to keep runs idempotent
        if dest.exists():
            logger.info(f"Skipping existing file (already on disk): {dest}")
            return False

        try:
            import requests

            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                content_type = r.headers.get("Content-Type", "").lower()
                logger.info(f"Downloading {filename} from {url} (Content-Type: {content_type})")

                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

            logger.info(f"===> SUCCESS: Saved {filename}")
            return True
        except Exception as e:
            logger.error(f"DOWNLOAD ERROR: {filename} | {e}")
            return False

    def run(self) -> None:
        """Run all configured providers and persist their PDFs to disk."""
        logger.info("--- STARTING PRODUCTION ACQUISITION ---")

        total_downloads = 0
        for name, provider in self.providers.items():
            logger.info(f"Starting acquisition for provider '{name}'")

            papers = provider.fetch_recent_papers(config.search.limit)
            logger.info(f"Provider '{name}' returned {len(papers)} papers.")

            count = 0
            for paper in papers:
                filename = f"{name}_{paper.id}.pdf"
                if self.download_pdf(paper.download_url, filename):
                    count += 1
                    total_downloads += 1

            logger.info(f"Provider '{name}' phase complete. New files: {count}")

        logger.info(f"--- PIPELINE COMPLETE --- Total new files: {total_downloads}")

