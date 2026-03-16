import time
from datetime import datetime, timedelta
from typing import List

import requests

from src.core.interfaces import PaperProvider, PaperMetadata
from src.config.loader import config
from src.utils.logger import logger


class ChemRxivClient(PaperProvider):
    """
    Client for ChemRxiv content hosted on Figshare.

    It:
    - calls the Figshare `/v2/articles` endpoint with `search_for`
    - paginates in batches of up to 50 results
    - filters articles to the last `config.search.date_range_days` days
    - resolves each article ID to its primary PDF via `/{id}/files`
    - returns a list of `PaperMetadata` with PDF download URLs.
    """

    def __init__(self) -> None:
        # e.g. "https://api.figshare.com/v2/articles"
        self.base_url = config.api.chemrxiv_base_url
        self.headers = {"User-Agent": config.api.user_agent}

    def fetch_recent_papers(self, limit: int) -> List[PaperMetadata]:
        """Return up to `limit` recent ChemRxiv papers that match the search term."""
        days = config.search.date_range_days
        cutoff = datetime.utcnow() - timedelta(days=days)

        logger.info(
            f"ChemRxivClient: fetching articles for '{config.search.term}', "
            f"limit={limit}, last {days} days"
        )

        page_size = min(limit, 50)  # Figshare page_size max is 50
        page = 1
        recent: list[dict] = []

        # Paginate until we have `limit` recent articles or there are no more pages
        while len(recent) < limit:
            params = {
                "search_for": config.search.term,
                "page_size": page_size,
                "page": page,
                "order": "published_date",
                "order_direction": "desc",
            }

            try:
                res = requests.get(self.base_url, params=params, headers=self.headers, timeout=30)
                res.raise_for_status()
                articles = res.json()
            except Exception as e:
                logger.error(f"ChemRxivClient: article search failed on page {page}: {e}")
                break

            if not articles:
                break

            for art in articles:
                try:
                    pub_str = art.get("published_date")
                    if not pub_str:
                        continue
                    pub_date = datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                    if pub_date >= cutoff:
                        recent.append(art)
                        if len(recent) >= limit:
                            break
                except Exception:
                    continue

            if len(articles) < page_size:
                # Last page
                break

            page += 1

        logger.info(f"ChemRxivClient: {len(recent)} articles within the last {days} days.")

        paper_list: List[PaperMetadata] = []

        for art in recent[:limit]:
            article_id = art.get("id")
            if article_id is None:
                continue

            try:
                files_url = f"{self.base_url}/{article_id}/files"
                files_res = requests.get(files_url, headers=self.headers, timeout=30)
                files_res.raise_for_status()
                files = files_res.json()

                pdf = next(
                    (
                        f
                        for f in files
                        if f.get("name", "").lower().endswith(".pdf")
                        and "supp" not in f.get("name", "").lower()
                    ),
                    None,
                )
                if not pdf:
                    pdf = next(
                        (f for f in files if f.get("name", "").lower().endswith(".pdf")),
                        None,
                    )

                if not pdf or not pdf.get("download_url"):
                    continue

                paper_list.append(
                    PaperMetadata(
                        id=str(article_id),
                        title=art.get("title", ""),
                        download_url=pdf["download_url"],
                        doi=art.get("doi", ""),
                    )
                )
            except Exception as e:
                logger.error(f"ChemRxivClient: failed to process article {article_id}: {e}")
                continue
            finally:
                time.sleep(0.2)  # basic rate limiting to avoid hammering the API

        logger.info(f"ChemRxivClient: prepared {len(paper_list)} PDF download tasks.")
        return paper_list