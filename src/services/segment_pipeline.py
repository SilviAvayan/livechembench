"""
SegmentPipeline: Orchestrates batch segmentation of all PDFs in raw_papers/.

- Scans the configured raw_papers directory for all *.pdf files
- Skips papers that already have a full output for the current engine
  (idempotent re-runs; engines that produce a layout artifact additionally
  require ``layout/<paper_id>/layout.json``).
- Writes one <paper_id>.json per paper to segmented_papers/
- With paddle_layout_dual_vl (default) or paddle_vl, exports figure crops to
  ``paths.segmented_assets/<paper_id>/`` and layout JSON + overlay PNGs to
  ``paths.segmented_layout/<paper_id>/``.
- PaddleOCR-VL and standalone PP-DocLayoutV3 are loaded once per run
  (dual-VL engine).
- Prints a summary report on completion
"""

from __future__ import annotations

import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class _PdfTimeout(Exception):
    """Raised by SIGALRM when a single PDF exceeds the per-PDF time limit."""


def _alarm_handler(signum: int, frame: object) -> None:
    raise _PdfTimeout()

from src.config.loader import config
from src.services.segmenter import SegmentedPaper, segment_paper
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _format_duration(seconds: float) -> str:
    """Human-readable duration; always includes exact seconds in parentheses."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(seconds)
    m, s = divmod(whole, 60)
    if m < 60:
        return f"{m}m {s}s ({seconds:.1f}s)"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s ({seconds:.1f}s)"


class SegmentPipeline:
    """Batch PDF segmentation (paddle_layout_dual_vl, paddle_vl, or Docling)."""

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
        self.layout_root = Path(config.paths.segmented_layout)
        if not self.layout_root.is_absolute():
            self.layout_root = _REPO_ROOT / self.layout_root
        self.layout_root.mkdir(parents=True, exist_ok=True)
        self.cfg = config.segmentation
        self._paddle_pipeline: Optional[Any] = None
        self._layout_model: Optional[Any] = None
        self._paddle_load_error: Optional[Exception] = None

    def _load_paddle_vl_pipeline(self) -> Any:
        """Single PaddleOCR-VL instance for the whole batch (~2GB once per run)."""
        from paddleocr import PaddleOCRVL

        pcfg = self.cfg.paddle_vl
        kwargs: Dict[str, Any] = {"pipeline_version": pcfg.pipeline_version}
        if pcfg.device:
            kwargs["device"] = pcfg.device
        logger.info(
            "Loading PaddleOCR-VL (legacy single-call) into memory (once per run)..."
        )
        return PaddleOCRVL(**kwargs)

    def _load_dual_vl_pipeline(self) -> Any:
        """PaddleOCR-VL-1.5 doc-parser pipeline (shared across the batch)."""
        from src.services.region_parser import create_vl_pipeline

        logger.info(
            "Loading PaddleOCR-VL-1.5 for paddle_layout_dual_vl (once per run)..."
        )
        return create_vl_pipeline(
            self.cfg.paddle_layout_dual_vl, self.cfg.pp_doclayout
        )

    def _load_layout_model(self) -> Any:
        """Standalone PP-DocLayoutV3 PaddleX model (Stage 1 of dual-VL engine)."""
        from src.services.layout_detector import create_pp_doclayout_model

        logger.info(
            "Loading standalone PP-DocLayoutV3 layout model (once per run)..."
        )
        return create_pp_doclayout_model(self.cfg.pp_doclayout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _output_path(self, paper_id: str) -> Path:
        return self.out_path / f"{paper_id}.json"

    def _already_done(self, paper_id: str) -> bool:
        if not self._output_path(paper_id).exists():
            return False
        if self.cfg.engine in ("paddle_layout_dual_vl", "paddle_vl"):
            layout_json = self.layout_root / paper_id / "layout.json"
            return layout_json.exists()
        return True

    def _save(self, paper: SegmentedPaper) -> None:
        path = self._output_path(paper.paper_id)
        with open(path, "w", encoding="utf-8") as f:
            f.write(paper.to_json())

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_batched(
        self,
        batch_size: int,
        offset: int = 0,
        limit: Optional[int] = None,
        per_pdf_timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run segmentation over all PDFs in raw_papers/ in sequential batches.

        Discovers the full sorted PDF list, applies ``offset`` and ``limit`` to
        define the universe of files to process, then iterates through them in
        chunks of ``batch_size``, calling :meth:`run` for each chunk.

        Returns an aggregated summary dict (``ok``, totals, per-batch breakdowns).
        """
        pdf_files = sorted(self.raw_path.glob("*.pdf")) + sorted(
            self.raw_path.glob("*.PDF")
        )
        total_available = len(pdf_files)

        if offset > 0:
            pdf_files = pdf_files[offset:]
        if limit is not None:
            pdf_files = pdf_files[:limit]

        universe = len(pdf_files)
        if universe == 0:
            logger.warning("run_batched: no PDFs in the specified universe; nothing to do.")
            return {"ok": True, "phase": "segmentation", "total": 0, "batches": []}

        n_batches = (universe + batch_size - 1) // batch_size
        logger.info(
            "=== BATCHED SEGMENTATION: %d PDF(s) → %d batch(es) of up to %d ===",
            universe,
            n_batches,
            batch_size,
        )

        aggregated: Dict[str, Any] = {
            "ok": True,
            "phase": "segmentation",
            "batch_size": batch_size,
            "total_available": total_available,
            "universe": universe,
            "success": 0,
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "timed_out": 0,
            "total_raw_chars": 0,
            "total_compressed_chars": 0,
            "batches": [],
        }

        for batch_idx in range(n_batches):
            batch_offset = offset + batch_idx * batch_size
            logger.info(
                "--- Batch %d/%d (offset=%d, size=%d) ---",
                batch_idx + 1,
                n_batches,
                batch_offset,
                batch_size,
            )
            result = self.run(
                limit=batch_size,
                offset=batch_offset,
                per_pdf_timeout=per_pdf_timeout,
            )
            aggregated["batches"].append(result)

            if not result.get("ok", True):
                aggregated["ok"] = False

            for key in ("success", "partial", "failed", "skipped", "timed_out"):
                aggregated[key] = aggregated.get(key, 0) + result.get(key, 0)
            aggregated["total_raw_chars"] += result.get("total_raw_chars", 0)
            aggregated["total_compressed_chars"] += result.get("total_compressed_chars", 0)

        logger.info(
            "=== BATCHED SEGMENTATION COMPLETE: %d success, %d partial, "
            "%d failed, %d timed_out, %d skipped across %d batch(es) ===",
            aggregated["success"],
            aggregated["partial"],
            aggregated["failed"],
            aggregated["timed_out"],

            aggregated["skipped"],
            n_batches,
        )
        return aggregated

    def run(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        per_pdf_timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run segmentation over PDFs in raw_papers/.

        PDFs are sorted by path. Apply ``offset`` then ``limit`` for batching, e.g.
        batch of 10: ``--offset 0 --limit 10``, next batch: ``--offset 10 --limit 10``.

        Returns a JSON-serializable summary dict (also written to ``_summary.json``
        when at least one PDF was eligible for the batch window). On validation
        failure, returns ``{"ok": False, ...}``.
        """
        pdf_files = sorted(self.raw_path.glob("*.pdf")) + sorted(self.raw_path.glob("*.PDF"))
        total_available = len(pdf_files)

        if offset < 0:
            logger.error("--offset must be non-negative.")
            return {"ok": False, "phase": "validation", "error": "invalid_offset"}

        if offset > 0 and offset >= total_available:
            logger.warning(
                f"--offset {offset} is past the end ({total_available} PDFs); nothing to do."
            )
            return {
                "ok": True,
                "phase": "segmentation",
                "total": 0,
                "reason": "offset_past_end",
            }

        pdf_files = pdf_files[offset:]

        if limit is not None:
            if limit < 1:
                logger.error("--limit must be a positive integer.")
                return {"ok": False, "phase": "validation", "error": "invalid_limit"}
            pdf_files = pdf_files[:limit]

        total = len(pdf_files)
        batch_start = offset + 1 if total else 0
        batch_end = offset + total

        if total > 0:
            logger.info(
                f"Batch window: sorted PDFs [{batch_start}..{batch_end}] "
                f"({total} file(s) in this run"
                + (f", offset={offset}" if offset else "")
                + (f", limit={limit}" if limit is not None else "")
                + f", {total_available} total in folder)."
            )

        if total == 0:
            if total_available == 0:
                logger.warning(f"No PDF files found in {self.raw_path}")
            else:
                logger.warning(
                    f"No PDFs in this batch window (offset={offset}, limit={limit}, "
                    f"{total_available} total in folder)."
                )
            return {
                "ok": True,
                "phase": "segmentation",
                "total": 0,
                "reason": "no_pdfs_in_window",
            }

        started_at = datetime.now(timezone.utc).astimezone()
        t_start = time.perf_counter()
        logger.info(f"=== SEGMENTATION PIPELINE STARTING ===")
        logger.info(f"Wall clock start: {started_at.isoformat(timespec='seconds')}")
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
        logger.info(f"Layout directory: {self.layout_root}")

        paddle_pipeline = self._paddle_pipeline
        layout_model = self._layout_model
        paddle_load_error = self._paddle_load_error

        if per_pdf_timeout:
            logger.info(f"Per-PDF timeout: {_format_duration(per_pdf_timeout)}")

        stats: Dict[str, int] = {
            "success": 0,
            "partial": 0,
            "failed": 0,
            "skipped": 0,
            "timed_out": 0,
        }
        total_raw_chars = 0
        total_compressed_chars = 0
        processed_wall_times: list[float] = []

        for idx, pdf_path in enumerate(pdf_files, start=1):
            paper_id = pdf_path.stem
            logger.info(f"[{idx}/{total}] Processing: {pdf_path.name}")

            # Skip if already done
            if self._already_done(paper_id):
                logger.info(f"  → Already segmented, skipping.")
                stats["skipped"] += 1
                continue

            engine = self.cfg.engine
            needs_paddle = engine in ("paddle_layout_dual_vl", "paddle_vl")
            needs_layout = engine == "paddle_layout_dual_vl"

            if needs_paddle and paddle_pipeline is None:
                if paddle_load_error is not None:
                    logger.error(
                        "Skipping remaining PDFs: Paddle pipeline failed to load earlier."
                    )
                    stats["failed"] += 1
                    continue
                try:
                    if engine == "paddle_layout_dual_vl":
                        paddle_pipeline = self._load_dual_vl_pipeline()
                    else:
                        paddle_pipeline = self._load_paddle_vl_pipeline()
                    self._paddle_pipeline = paddle_pipeline
                except Exception as exc:
                    paddle_load_error = exc
                    self._paddle_load_error = exc
                    logger.error(
                        "Could not initialize PaddleOCR-VL: %s. "
                        "Fix the install or set segmentation.engine to docling.",
                        exc,
                    )
                    stats["failed"] += 1
                    continue

            if needs_layout and layout_model is None:
                try:
                    layout_model = self._load_layout_model()
                    self._layout_model = layout_model
                except Exception as exc:
                    paddle_load_error = exc
                    self._paddle_load_error = exc
                    logger.error(
                        "Could not initialize PP-DocLayoutV3 standalone model: %s. "
                        "Fix the install or switch segmentation.engine.",
                        exc,
                    )
                    stats["failed"] += 1
                    continue

            try:
                t0 = time.perf_counter()
                if per_pdf_timeout:
                    signal.signal(signal.SIGALRM, _alarm_handler)
                    signal.alarm(per_pdf_timeout)
                try:
                    paper = segment_paper(
                        pdf_path,
                        self.cfg,
                        self.assets_root,
                        paddle_pipeline=paddle_pipeline,
                    )
                    self._save(paper)
                finally:
                    if per_pdf_timeout:
                        signal.alarm(0)

                wall = time.perf_counter() - t0
                processed_wall_times.append(wall)

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
                logger.info(f"  → Time this PDF: {_format_duration(wall)}")

            except _PdfTimeout:
                wall = time.perf_counter() - t0
                logger.warning(
                    f"  → TIMEOUT after {_format_duration(wall)}: {pdf_path.name} "
                    f"exceeded the {_format_duration(per_pdf_timeout)} limit. "
                    "Skipping and resetting pipeline for next PDF."
                )
                stats["timed_out"] += 1
                # Discard the pipeline — its internal VLM worker thread is in an
                # unknown state; the next PDF will get a fresh model load.
                paddle_pipeline = None
                self._paddle_pipeline = None
                layout_model = None
                self._layout_model = None

            except Exception as exc:
                logger.error(f"  → UNEXPECTED ERROR for {pdf_path.name}: {exc}")
                stats["failed"] += 1

        elapsed = time.perf_counter() - t_start
        finished_at = datetime.now(timezone.utc).astimezone()

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
        logger.info(f"  Wall clock end:    {finished_at.isoformat(timespec='seconds')}")
        logger.info(f"  Total papers:      {total}")
        logger.info(f"  Success:           {stats['success']}")
        logger.info(f"  Partial:           {stats['partial']}")
        logger.info(f"  Failed:            {stats['failed']}")
        logger.info(f"  Timed out:         {stats['timed_out']}")
        logger.info(f"  Skipped (cached):  {stats['skipped']}")
        logger.info(f"  Total raw chars:   {total_raw_chars:,}")
        logger.info(f"  Compressed chars:  {total_compressed_chars:,}")
        logger.info(f"  Compression ratio: {overall_ratio}")
        logger.info(f"  Elapsed time:      {_format_duration(elapsed)}")
        if processed_wall_times:
            avg = sum(processed_wall_times) / len(processed_wall_times)
            logger.info(
                f"  Processed PDFs:    {len(processed_wall_times)} "
                f"(avg {_format_duration(avg)} per PDF, incl. first model load on first PDF)"
            )
        logger.info("=" * 60)

        # Write a machine-readable summary JSON
        summary = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "offset": offset,
            "limit": limit,
            "per_pdf_timeout": per_pdf_timeout,
            "total": total,
            "success": stats["success"],
            "partial": stats["partial"],
            "failed": stats["failed"],
            "timed_out": stats["timed_out"],
            "skipped": stats["skipped"],
            "total_raw_chars": total_raw_chars,
            "total_compressed_chars": total_compressed_chars,
            "compression_ratio": overall_ratio,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_human": _format_duration(elapsed),
        }
        if processed_wall_times:
            summary["pdf_times_seconds"] = [round(x, 2) for x in processed_wall_times]
            summary["avg_pdf_seconds"] = round(
                sum(processed_wall_times) / len(processed_wall_times), 2
            )
        summary_path = self.out_path / "_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary written to {summary_path}")
        summary["ok"] = True
        summary["phase"] = "segmentation"
        return summary
