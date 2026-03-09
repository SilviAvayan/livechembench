import requests
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from src.config.loader import config

logger = logging.getLogger("ChemistryProject")

class DownloadPipeline:
    def __init__(self, provider_name=None):
        """
        Initialize the pipeline. 
        Accepts provider_name to maintain compatibility with main.py.
        """
        # Handles potential Pydantic model vs Dictionary access
        try:
            self.raw_path = Path(config.paths.raw_papers)
        except (AttributeError, TypeError):
            self.raw_path = Path(config.paths["raw_papers"])
            
        self.raw_path.mkdir(parents=True, exist_ok=True)
        
        # Use a standard Browser User-Agent to prevent 403 Forbidden errors
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Define 30-day window
        self.limit = config.search.limit
        self.delta_days = config.search.date_range_days
        self.since_date_figshare = (datetime.now() - timedelta(days=self.delta_days)).strftime("%Y-%m-%d")
        self.start_date_pmc = (datetime.now() - timedelta(days=self.delta_days)).strftime("%Y/%m/%d")
        self.end_date_pmc = datetime.now().strftime("%Y/%m/%d")

    def download_file(self, url, filename):
        """Downloads binary content with stream handling for production stability."""
        try:
            path = self.raw_path / filename
            if path.exists():
                return False
            
            with requests.get(url, headers=self.headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"===> DOWNLOADED: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False

    def fetch_chemrxiv(self):
        """Targets the official ChemRxiv repository (Institution 743) on Figshare."""
        logger.info(f"Fetching ChemRxiv preprints since {self.since_date_figshare}...")
        
        search_url = "https://api.figshare.com/v2/articles/search"
        payload = {
            "published_since": self.since_date_figshare,
            "institution": 743, # Dedicated ChemRxiv Institution ID
            "item_type": 3,      # Preprints only
            "limit": self.limit,
            "order": "published_date",
            "order_direction": "desc"
        }
        
        try:
            resp = requests.post(search_url, json=payload, headers=self.headers)
            resp.raise_for_status()
            papers = resp.json()
            logger.info(f"Found {len(papers)} ChemRxiv records. Extracting PDFs...")

            for p in papers:
                article_id = p['id']
                # Secondary call to get file URLs
                detail = requests.get(f"https://api.figshare.com/v2/articles/{article_id}", headers=self.headers).json()
                
                files = detail.get('files', [])
                # Prioritize main manuscript over supplementary
                pdf_files = [f for f in files if f['name'].lower().endswith('.pdf')]
                main_pdf = next((f for f in pdf_files if not any(x in f['name'].lower() for x in ["supp", "si", "check"])), None)
                
                target = main_pdf or (pdf_files[0] if pdf_files else None)
                
                if target:
                    self.download_file(target['download_url'], f"chemrxiv_{article_id}.pdf")
                    time.sleep(0.5) # Polite delay
        except Exception as e:
            logger.error(f"ChemRxiv Pipeline Error: {e}")

    def fetch_pubmed_central(self):
        """Targets PubMed Central using the official E-Utils and OA Web Service."""
        logger.info(f"Fetching Biology papers from PMC between {self.start_date_pmc} and {self.end_date_pmc}...")
        
        # Step 1: Search for IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pmc",
            "term": f"biology[filter] AND open access[filter] AND (\"{self.start_date_pmc}\"[PubDate] : \"{self.end_date_pmc}\"[PubDate])",
            "retmode": "json",
            "retmax": 15 # Production limit to avoid throttling
        }
        
        try:
            search_res = requests.get(search_url, params=params).json()
            pmcids = search_res.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmcids)} Biology papers in PMC. Resolving PDF links...")

            for pmcid in pmcids:
                # Step 2: Get Open Access download link from OA Service
                oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC{pmcid}&format=json"
                oa_res = requests.get(oa_url).json()
                records = oa_res.get("oa", {}).get("record", [])
                
                if records:
                    links = records[0].get("link", [])
                    pdf_link = next((l['href'] for l in links if l.get('format') == 'pdf'), None)
                    
                    if pdf_link:
                        # Required by NCBI: 1-second delay between requests to avoid 403
                        time.sleep(1.0)
                        self.download_file(pdf_link, f"pmc_{pmcid}.pdf")
                    else:
                        logger.warning(f"PMC{pmcid} has no PDF format available in OA service.")
        except Exception as e:
            logger.error(f"PubMed Central Pipeline Error: {e}")

    def run(self):
        logger.info("--- Starting Production-Grade Data Acquisition ---")
        self.fetch_chemrxiv()
        self.fetch_pubmed_central()
        logger.info("--- Data Acquisition Complete ---")