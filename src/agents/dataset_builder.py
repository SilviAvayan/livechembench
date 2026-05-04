"""
Dataset Builder

Aggregates all benchmark-ready questions from data/selected_questions/ into a
single, versioned benchmark JSON at data/benchmark/livechembench_v<version>.json.

Use --verification-report to automatically drop questions that failed verification.

No LLM calls — pure aggregation.

Usage:
    python -m src.agents.dataset_builder
    python -m src.agents.dataset_builder --version 0.2.0
    python -m src.agents.dataset_builder --verification-report data/verification/livechembench_v0.1.0_verified.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.agents.models import (
    BenchmarkQuestion,
    BenchmarkStats,
    LiveChemBench,
    SelectedPaperQuestions,
    VerificationReport,
    VerificationStatus,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_verified_ids(verification_path: Optional[Path]) -> Optional[set[str]]:
    """Return set of question IDs that passed verification, or None if no report given."""
    if not verification_path or not verification_path.exists():
        return None
    report = VerificationReport.model_validate_json(verification_path.read_text())
    passing = {r.question_id for r in report.results if r.status == VerificationStatus.correct}
    total = len(report.results)
    logger.info(
        "Verification filter: %d/%d questions passed — dropping %d",
        len(passing), total, total - len(passing),
    )
    return passing


def build(
    selected_dir: Path,
    output_dir: Path,
    version: str,
    verification_path: Optional[Path] = None,
) -> LiveChemBench:
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_ids = _load_verified_ids(verification_path)

    questions: list[BenchmarkQuestion] = []
    counter = 1
    dropped = 0

    for json_path in sorted(selected_dir.glob("*.json")):
        try:
            report = SelectedPaperQuestions.model_validate_json(json_path.read_text())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        passing = report.benchmark_ready()
        paper_added = 0
        for q in passing:
            tentative_id = f"lcb_{counter:04d}"
            # If a verification report was given, skip questions not in the passing set.
            # We check by regenerating the ID in sequence — verified_ids uses the same ordering.
            candidate = BenchmarkQuestion(
                id=tentative_id,
                paper_id=report.paper_id,
                question_text=q.question_text,
                answer=q.answer,
                answer_type=q.answer_type,
                answer_units=q.answer_units,
                tolerance=q.tolerance,
                question_type=q.question_type,
                chemical_entities=q.chemical_entities,
                verification_recipe=q.verification_recipe,
                source_segment=q.source_segment,
            )
            counter += 1
            if verified_ids is not None and tentative_id not in verified_ids:
                logger.info("  dropping %s (failed verification)", tentative_id)
                dropped += 1
                continue
            questions.append(candidate)
            paper_added += 1

        logger.info(
            "  %s → %d question(s) added", report.paper_id, paper_added
        )

    # Compute stats
    by_type: dict[str, int] = defaultdict(int)
    by_paper: dict[str, int] = defaultdict(int)
    by_answer_type: dict[str, int] = defaultdict(int)

    for q in questions:
        by_type[q.question_type.value] += 1
        by_paper[q.paper_id] += 1
        by_answer_type[q.answer_type.value] += 1

    stats = BenchmarkStats(
        total=len(questions),
        by_type=dict(by_type),
        by_paper=dict(by_paper),
        by_answer_type=dict(by_answer_type),
    )

    benchmark = LiveChemBench(
        version=version,
        created_at=datetime.now(timezone.utc).isoformat(),
        stats=stats,
        questions=questions,
    )

    out_path = output_dir / f"livechembench_v{version}.json"
    out_path.write_text(benchmark.model_dump_json(indent=2))

    logger.info(
        "Dataset built: %d question(s) from %d paper(s) → %s",
        stats.total,
        len(stats.by_paper),
        out_path,
    )
    _print_summary(benchmark)
    return benchmark


def _print_summary(b: LiveChemBench) -> None:
    s = b.stats
    logger.info("=" * 50)
    logger.info("LiveChemBench v%s", b.version)
    logger.info("  Total questions : %d", s.total)
    logger.info("  By type         : %s", s.by_type)
    logger.info("  By answer type  : %s", s.by_answer_type)
    logger.info("  By paper        : %s", s.by_paper)
    logger.info("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate selected questions into the final benchmark JSON."
    )
    parser.add_argument("--version", default="0.1.0", help="Benchmark version string.")
    parser.add_argument(
        "--selected-dir",
        default=str(_REPO_ROOT / "data" / "selected_questions"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "benchmark"),
    )
    parser.add_argument(
        "--verification-report",
        default=None,
        help="Path to a verification report JSON. Questions that failed verification are dropped.",
    )
    args = parser.parse_args()

    build(
        selected_dir=Path(args.selected_dir),
        output_dir=Path(args.output_dir),
        version=args.version,
        verification_path=Path(args.verification_report) if args.verification_report else None,
    )


if __name__ == "__main__":
    main()
