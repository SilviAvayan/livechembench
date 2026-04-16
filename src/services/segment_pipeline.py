"""
SegmentPipeline: Orchestrates batch segmentation of all PDFs in raw_papers/.

- Scans the configured raw_papers directory for all *.pdf files
- Skips already-segmented papers (idempotent re-runs)
- Writes one <paper_id>.json per paper to segmented_papers/
- Prints a summary report on completion
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from src.config.loader import config
from src.services.segmenter import SegmentedPaper, segment_paper
from src.utils.logger import logger


class SegmentPipeline:
    """Batch segmentation of all PDFs using Docling."""

    def __init__(self) -> None:
        self.raw_path = Path(config.paths.raw_papers)
        self.out_path = Path(config.paths.segmented_papers)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.cfg = config.segmentation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _output_path(self, paper_id: str) -> Path:
        return self.out_path / f"{paper_id}.json"

    def _already_done(self, paper_id: str) -> bool:
        return self._output_path(paper_id).exists()

    def _save(self, paper: SegmentedPaper) -> None:
        path = self._output_path(paper.paper_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(paper.to_json())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run segmentation over all PDFs in raw_papers/."""
        pdf_files = sorted(self.raw_path.glob("*.pdf")) + sorted(self.raw_path.glob("*.PDF"))
        total = len(pdf_files)

        if total == 0:
            logger.warning(f"No PDF files found in {self.raw_path}")
            return

        logger.info(f"=== SEGMENTATION PIPELINE STARTING ===")
        logger.info(f"Found {total} PDF files in {self.raw_path}")
        logger.info(f"Output directory: {self.out_path}")

        stats: Dict[str, int] = {"success": 0, "partial": 0, "failed": 0, "skipped": 0}
        total_raw_chars = 0
        total_compressed_chars = 0
        start_time = time.time()

        for idx, pdf_path in enumerate(pdf_files, start=1):
            paper_id = pdf_path.stem
            logger.info(f"[{idx}/{total}] Processing: {pdf_path.name}")

            # Skip if already done
            if self._already_done(paper_id):
                logger.info(f"  → Already segmented, skipping.")
                stats["skipped"] += 1
                continue

            try:
                paper = segment_paper(pdf_path, self.cfg)
                self._save(paper)

                stats[paper.extraction_status] = stats.get(paper.extraction_status, 0) + 1
                total_raw_chars += paper.raw_char_count
                total_compressed_chars += paper.compressed_char_count

                ratio = (
                    f"{100 * paper.compressed_char_count / paper.raw_char_count:.1f}%"
                    if paper.raw_char_count > 0
                    else "N/A"
                )
                logger.info(
                    f"  → [{paper.extraction_status.upper()}] "
                    f"sections={paper.section_count}, "
                    f"raw={paper.raw_char_count:,} chars, "
                    f"compressed={paper.compressed_char_count:,} chars "
                    f"({ratio} of original)"
                )

            except Exception as exc:
                logger.error(f"  → UNEXPECTED ERROR for {pdf_path.name}: {exc}")
                stats["failed"] += 1

        elapsed = time.time() - start_time

        # ---- Summary report -----------------------------------------------
        overall_ratio = (
            f"{100 * total_compressed_chars / total_raw_chars:.1f}%"
            if total_raw_chars > 0
            else "N/A"
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("SEGMENTATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Total papers:      {total}")
        logger.info(f"  Success:           {stats['success']}")
        logger.info(f"  Partial:           {stats['partial']}")
        logger.info(f"  Failed:            {stats['failed']}")
        logger.info(f"  Skipped (cached):  {stats['skipped']}")
        logger.info(f"  Total raw chars:   {total_raw_chars:,}")
        logger.info(f"  Compressed chars:  {total_compressed_chars:,}")
        logger.info(f"  Compression ratio: {overall_ratio}")
        logger.info(f"  Elapsed time:      {elapsed:.1f}s")
        logger.info("=" * 60)

        # Write a machine-readable summary JSON
        summary = {
            "total": total,
            "success": stats["success"],
            "partial": stats["partial"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "total_raw_chars": total_raw_chars,
            "total_compressed_chars": total_compressed_chars,
            "compression_ratio": overall_ratio,
            "elapsed_seconds": round(elapsed, 1),
        }
        summary_path = self.out_path / "_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary written to {summary_path}")
