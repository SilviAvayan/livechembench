"""
SegmentPipeline: Orchestrates batch segmentation of all PDFs in raw_papers/.

- Scans the configured raw_papers directory for all *.pdf files
- Skips already-segmented papers (idempotent re-runs)
- Writes one <paper_id>.json per paper to segmented_papers/
- With engine paddle_vl, exports figure crops under paths.segmented_assets/<paper_id>/
- Prints a summary report on completion
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.loader import config
from src.services.segmenter import SegmentedPaper, segment_paper
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SegmentPipeline:
    """Batch PDF segmentation (PaddleOCR-VL-1.5 or Docling per config)."""

    def __init__(self) -> None:
        self.raw_path = Path(config.paths.raw_papers)
        if not self.raw_path.is_absolute():
            self.raw_path = _REPO_ROOT / self.raw_path
        self.out_path = Path(config.paths.segmented_papers)
        if not self.out_path.is_absolute():
            self.out_path = _REPO_ROOT / self.out_path
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.assets_root = Path(config.paths.segmented_assets)
        if not self.assets_root.is_absolute():
            self.assets_root = _REPO_ROOT / self.assets_root
        self.assets_root.mkdir(parents=True, exist_ok=True)
        self.cfg = config.segmentation

    def _load_paddle_pipeline(self) -> Any:
        """Single PaddleOCR-VL instance for the whole batch (avoids reloading ~2GB per PDF)."""
        from paddleocr import PaddleOCRVL

        pcfg = self.cfg.paddle_vl
        kwargs: Dict[str, Any] = {"pipeline_version": pcfg.pipeline_version}
        if pcfg.device:
            kwargs["device"] = pcfg.device
        logger.info("Loading PaddleOCR-VL models into memory (once per run)...")
        return PaddleOCRVL(**kwargs)

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

    def run(self, limit: Optional[int] = None) -> None:
        """Run segmentation over PDFs in raw_papers/.

        :param limit: If set, only the first ``limit`` PDFs (sorted by path) are
            considered. Useful for smoke tests without moving files.
        """
        pdf_files = sorted(self.raw_path.glob("*.pdf")) + sorted(self.raw_path.glob("*.PDF"))
        total_available = len(pdf_files)

        if limit is not None:
            if limit < 1:
                logger.error("--limit must be a positive integer.")
                return
            pdf_files = pdf_files[:limit]
            logger.info(
                f"Limit active: processing {len(pdf_files)} PDF(s) "
                f"(first {limit} in sorted order, {total_available} total in folder)."
            )

        total = len(pdf_files)

        if total == 0:
            logger.warning(f"No PDF files found in {self.raw_path}")
            return

        logger.info(f"=== SEGMENTATION PIPELINE STARTING ===")
        logger.info(f"Engine: {self.cfg.engine}")
        logger.info(
            f"Found {total} PDF file(s) in this run"
            + (
                f" ({total_available} total in {self.raw_path})"
                if total_available != total
                else f" in {self.raw_path}"
            )
        )
        logger.info(f"Output directory: {self.out_path}")
        logger.info(f"Assets directory: {self.assets_root}")

        paddle_pipeline: Optional[Any] = None
        paddle_load_error: Optional[Exception] = None

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

            if self.cfg.engine == "paddle_vl" and paddle_pipeline is None:
                if paddle_load_error is not None:
                    logger.error(
                        "Skipping remaining PDFs: PaddleOCR-VL failed to load earlier."
                    )
                    stats["failed"] += 1
                    continue
                try:
                    paddle_pipeline = self._load_paddle_pipeline()
                except Exception as exc:
                    paddle_load_error = exc
                    logger.error(
                        "Could not initialize PaddleOCR-VL: %s. "
                        "Fix the install or set segmentation.engine to docling.",
                        exc,
                    )
                    stats["failed"] += 1
                    continue

            try:
                paper = segment_paper(
                    pdf_path,
                    self.cfg,
                    self.assets_root,
                    paddle_pipeline=paddle_pipeline,
                )
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
