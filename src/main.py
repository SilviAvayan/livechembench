from src.services.pipeline import DownloadPipeline
from src.utils.logger import logger

def main():
    try:
        pipeline = DownloadPipeline()
        pipeline.run()
    except Exception as e:
        logger.error(f"Application crashed: {e}")

if __name__ == "__main__":
    main()