import argparse

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="LiveChemBench — paper acquisition, segmentation, and agent pipeline."
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
    parser.add_argument(
        "--agents",
        action="store_true",
        help=(
            "Run the full agent pipeline: novelty selection → entity extraction → "
            "question proposal → tri-critic verification → dataset build → evaluation."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="With --agents: build the dataset but skip the evaluator step.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="With --segment, only process the first N PDFs (sorted by path). Omit to process all.",
    )
    args = parser.parse_args()

    try:
        if args.agents:
            from src.config.loader import config
            from src.pipeline.agent_pipeline import AgentPipeline
            pipeline = AgentPipeline(config)
            pipeline.run_sync(skip_eval=args.skip_eval)
        elif args.segment:
            from src.services.segment_pipeline import SegmentPipeline
            pipeline = SegmentPipeline()
            pipeline.run(limit=args.limit)
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