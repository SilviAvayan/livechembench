import aiohttp
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFDownloader:

    async def download_pdf(self, url: str, path: Path):

        path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:

                with open(path, "wb") as f:
                    f.write(await resp.read())

        logger.info(f"Downloaded {path.name}")