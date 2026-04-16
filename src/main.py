import argparse

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="LiveChemBench — paper acquisition and segmentation."
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run the segmentation pipeline on all raw PDFs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run the download pipeline to fetch new papers (default action).",
    )
    args = parser.parse_args()

    try:
        if args.segment:
            from src.services.segment_pipeline import SegmentPipeline
            pipeline = SegmentPipeline()
            pipeline.run()
        else:
            # Default: run download pipeline
            from src.services.pipeline import DownloadPipeline
            pipeline = DownloadPipeline()
            pipeline.run()
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        raise


if __name__ == "__main__":
    main()