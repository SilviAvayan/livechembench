import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.services.pipeline import DownloadPipeline
from src.utils.logger import logger
from src.config.loader import config

def main():
    print(config.search)   # <-- debug line

    try:
        pipeline = DownloadPipeline(provider_name="chemrxiv")
        pipeline.run()
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()