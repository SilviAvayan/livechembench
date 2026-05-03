import argparse
import sys

from src.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="LiveChemBench — paper acquisition and segmentation."
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run the agent orchestrator (multi-step workflows; see --workflow).",
    )
    parser.add_argument(
        "--workflow",
        choices=["ingest_then_segment", "download_only", "segment_only"],
        default="ingest_then_segment",
        help="With --agent, which preset workflow to execute.",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run the segmentation pipeline directly (single step).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Only fetch papers (ignored when --segment or --agent is set).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="With --segment or agent segment step: max PDFs after --offset. Omit for no cap.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="K",
        help="With --segment or agent segment step: skip first K sorted PDFs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="B",
        help=(
            "With --segment: split the run into sequential batches of B PDFs each. "
            "Combines with --offset and --limit to narrow the universe first. "
            "Omit to process all eligible PDFs in a single pass."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        metavar="MINUTES",
        help=(
            "With --segment: per-PDF wall-clock timeout in minutes. "
            "A PDF that exceeds this limit is skipped and the pipeline is reset "
            "for the next file. Omit to disable the timeout."
        ),
    )
    args = parser.parse_args()

    try:
        if args.agent:
            from src.agent.orchestrator import AgentOrchestrator
            from src.agent.presets import (
                download_only,
                ingest_then_segment,
                segment_only,
            )
            from src.agent.types import RunStatus
            from src.config.loader import config

            if args.workflow == "ingest_then_segment":
                wf = ingest_then_segment(
                    segment_limit=args.limit,
                    segment_offset=args.offset,
                )
            elif args.workflow == "download_only":
                wf = download_only()
            else:
                wf = segment_only(limit=args.limit, offset=args.offset)

            result = AgentOrchestrator(config).run(wf)
            logger.info("Orchestration %s (run_id=%s)", result.status.value, result.run_id)
            sys.exit(0 if result.status is RunStatus.COMPLETED else 1)

        if args.segment:
            from src.services.segment_pipeline import SegmentPipeline

            timeout_secs = args.timeout * 60 if args.timeout else None
            pipeline = SegmentPipeline()
            if args.batch_size is not None:
                if args.batch_size < 1:
                    logger.error("--batch-size must be a positive integer.")
                    sys.exit(1)
                pipeline.run_batched(
                    batch_size=args.batch_size,
                    offset=args.offset,
                    limit=args.limit,
                    per_pdf_timeout=timeout_secs,
                )
            else:
                pipeline.run(
                    limit=args.limit,
                    offset=args.offset,
                    per_pdf_timeout=timeout_secs,
                )
            return

        from src.services.pipeline import DownloadPipeline

        DownloadPipeline().run()
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        raise


if __name__ == "__main__":
    main()
