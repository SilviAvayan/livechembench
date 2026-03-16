import time
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
from typing import List

import requests

from src.config.loader import config
from src.core.interfaces import PaperProvider, PaperMetadata
from src.utils.logger import logger


class PubMedClient(PaperProvider):
    """
    Client for PubMed Central (PMC) via NCBI E-utilities + OA Web Service.

    It:
    - runs a PMC search using the configured boolean `config.search.term`
      combined with `open access[filter]`
    - resolves each PMC ID to a PDF link via the PMC OA Web Service
    - returns a list of `PaperMetadata` records with download URLs.
    """

    def __init__(self) -> None:
        self.esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        self.api_key = config.api.ncbi_api_key

    def fetch_recent_papers(self, limit: int) -> List[PaperMetadata]:
        """Return up to `limit` recent open-access PMC papers that match the search term."""
        days = config.search.date_range_days
        since = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")

        # Treat the configured search term as the full boolean query,
        # and only add an open-access filter.
        term = f"({config.search.term}) AND open access[filter]"
        logger.info(f"PubMedClient: searching PMC for '{term}' within last {days} days (since {since})")

        params = {
            "db": "pmc",
            "term": term,
            "reldate": days,
            "retmode": "json",
            "retmax": limit,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            res = requests.get(self.esearch_url, params=params, timeout=30)
            res.raise_for_status()
            payload = res.json()
            ids = payload.get("esearchresult", {}).get("idlist", [])
            logger.info(f"PubMedClient: PMC identified {len(ids)} candidate IDs.")
        except Exception as e:
            logger.error(f"PubMedClient: search failed: {e}")
            return []

        papers: List[PaperMetadata] = []

        for pmcid in ids:
            time.sleep(0.2)
            try:
                oa_params = {"id": f"PMC{pmcid}", "format": "pdf"}
                if self.api_key:
                    oa_params["api_key"] = self.api_key

                r = requests.get(self.oa_url, params=oa_params, timeout=30)
                r.raise_for_status()

                root = ET.fromstring(r.text)
                records = root.find("records")
                if records is None:
                    continue

                record = records.find("record")
                if record is None:
                    continue

                pdf_link = None
                for link in record.findall("link"):
                    if link.get("format") == "pdf":
                        pdf_link = link.get("href")
                        break

                if not pdf_link:
                    continue

                citation = record.get("citation", "")

                papers.append(
                    PaperMetadata(
                        id=str(pmcid),
                        title=citation,
                        download_url=pdf_link,
                        doi="",
                    )
                )
            except Exception as e:
                logger.error(f"PubMedClient: OA lookup failed for PMC{pmcid}: {e}")
                continue

        logger.info(f"PubMedClient: prepared {len(papers)} PDF download tasks.")
        return papers

