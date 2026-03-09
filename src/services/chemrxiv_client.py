import requests
from datetime import datetime, timedelta
from typing import List
from src.core.interfaces import PaperProvider, PaperMetadata
from src.config.loader import config
from src.utils.logger import logger
import time 



class ChemRxivClient(PaperProvider):
    def __init__(self):
        self.base_url = config.api.chemrxiv_base_url
        self.headers = {"User-Agent": config.api.user_agent}

    def fetch_recent_papers(self):
        all_papers = []
        offset = 0
        page_size = 100  # Figshare max per request
        max_offset = 1000  # Figshare API limit

        since_date = (datetime.now() - timedelta(days=config.search.date_range_days)).strftime('%Y-%m-%d')
        logger.info(f"Searching for papers published since {since_date}")

        while offset <= max_offset:
            params = {
                "search_for": config.search.term,
                "offset": offset,
                "limit": page_size,
                "published_since": since_date
            }

            try:
                response = requests.get(self.base_url, params=params, headers=self.headers)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed: {e}")
                break

            papers_page = response.json()
            if not papers_page:
                break

            all_papers.extend(papers_page)
            logger.info(f"Fetched {len(papers_page)} papers, total so far: {len(all_papers)}")

            # Increment offset to fetch next batch
            offset += page_size

        logger.info(f"Total papers fetched: {len(all_papers)}")
        return all_papers   