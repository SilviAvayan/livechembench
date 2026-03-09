import aiohttp
from datetime import datetime, timedelta
from ..src.utils.logger import get_logger

logger = get_logger(__name__)


class ChemRxivClient:

    BASE_URL = "https://chemrxiv.org/engage/chemrxiv/public-api"

    async def get_last_month_papers(self):

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        url = f"{self.BASE_URL}/articles"

        params = {
            "limit": 100,
            "sort": "published",
            "order": "desc"
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        papers = []

        for item in data["items"]:
            papers.append({
                "id": item["id"],
                "title": item["title"],
                "pdf_url": item["assets"][0]["url"]
            })

        logger.info(f"Fetched {len(papers)} papers")

        return papers