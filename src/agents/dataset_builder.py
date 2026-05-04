"""
Dataset Builder

Aggregates all benchmark-ready questions from data/selected_questions/ into a
single, versioned benchmark JSON at data/benchmark/livechembench_v<version>.json.

Populates the new schema with:
  - Structured verifier (type + recipe) derived from question_type
  - Filters (ill_defined, missing_conditions, guessable) loaded from critiques
  - Provenance (month, paper_source, conversion_tool, pubchem_query_log_hash)
  - Primary PubChem CID from data/pubchem_links/

Use --verification-report to automatically drop questions that failed verification.

Usage:
    python -m src.agents.dataset_builder
    python -m src.agents.dataset_builder --version 0.3.0
    python -m src.agents.dataset_builder --version 0.3.0 --verification-report data/verification/...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.agents.models import (
    BenchmarkQuestion,
    BenchmarkStats,
    CandidateQuestion,
    Filters,
    LiveChemBench,
    PaperCritiqueReport,
    Provenance,
    QuestionType,
    SelectedPaperQuestions,
    Verifier,
    VerifierRecipe,
    VerificationReport,
    VerificationStatus,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_verified_ids(verification_path: Optional[Path]) -> Optional[set[str]]:
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


def _format_paper_id(raw_paper_id: str) -> str:
    """Convert internal paper_id to 'source:TYPE:ID' format."""
    # Already formatted
    if raw_paper_id.startswith("source:"):
        return raw_paper_id
    # e.g. "chemrxiv_30467288" or "pubmed_12478036" or "pmc_12987760"
    m = re.match(r"^([a-zA-Z]+)_(.+)$", raw_paper_id)
    if m:
        source_type, source_id = m.group(1).lower(), m.group(2)
        return f"source:{source_type}:{source_id}"
    # Legacy: "15078972_Some_Title" — extract leading digits as ID
    m2 = re.match(r"^(\d+)_", raw_paper_id)
    if m2:
        return f"source:unknown:{m2.group(1)}"
    return f"source:unknown:{raw_paper_id}"


def _parse_source_type(raw_paper_id: str) -> str:
    """Return just the source type (pubmed, chemrxiv, pmc, …)."""
    formatted = _format_paper_id(raw_paper_id)
    parts = formatted.split(":")
    return parts[1] if len(parts) >= 3 else "unknown"


def _build_verifier(q: CandidateQuestion) -> Verifier:
    """Derive a structured Verifier from the question's type and verification_recipe."""
    recipe_text = q.verification_recipe
    if q.question_type == QuestionType.T1:
        # PubChem property lookup — try to parse endpoint/field from recipe text
        endpoint = None
        field = None
        ep_m = re.search(r"endpoint[:\s]+([^\s,;]+)", recipe_text, re.I)
        if ep_m:
            endpoint = ep_m.group(1)
        f_m = re.search(r"field[:\s]+([^\s,;]+)", recipe_text, re.I)
        if f_m:
            field = f_m.group(1)
        return Verifier(
            type="pubchem_pugrest",
            recipe=VerifierRecipe(
                endpoint=endpoint,
                field=field,
                description=recipe_text,
            ),
        )
    elif q.question_type == QuestionType.T2:
        # RDKit computation — try to extract function name
        func = None
        fn_m = re.search(r"(rdMolDescriptors\.\w+|rdkit\.\w+|Chem\.\w+)", recipe_text, re.I)
        if fn_m:
            func = fn_m.group(1)
        return Verifier(
            type="rdkit",
            recipe=VerifierRecipe(
                function=func,
                input="smiles",
                description=recipe_text,
            ),
        )
    else:  # T3 — hybrid
        return Verifier(
            type="hybrid",
            recipe=VerifierRecipe(description=recipe_text),
        )


def _load_critiques(critiques_dir: Path, paper_id: str) -> Optional[PaperCritiqueReport]:
    path = critiques_dir / f"{paper_id}.json"
    if not path.exists():
        return None
    try:
        return PaperCritiqueReport.model_validate_json(path.read_text())
    except (ValidationError, json.JSONDecodeError):
        return None


def _build_filters(
    q: CandidateQuestion,
    critique_report: Optional[PaperCritiqueReport],
) -> Filters:
    """Build a Filters object by inspecting critic verdicts for this question."""
    ill_defined = False
    missing_conditions: list[str] = []
    # guessable — we know all surviving questions passed novelty (Critic 3);
    # this field stays False for benchmark_ready questions.

    if critique_report is None:
        return Filters()

    for record in critique_report.critiques:
        if record.question_text != q.question_text:
            continue
        if record.critic.value == "ill_defined":
            # FAIL means it was ill-defined; surviving questions have PASS or NEEDS_REPAIR that was fixed
            from src.agents.models import CriticVerdict
            ill_defined = record.result.verdict == CriticVerdict.fail
        elif record.critic.value == "missing_conditions":
            if record.result.missing_conditions:
                missing_conditions = record.result.missing_conditions

    return Filters(
        ill_defined=ill_defined,
        missing_conditions=missing_conditions,
        guessable=False,
    )


def _load_pubchem_links(pubchem_dir: Path, paper_id: str) -> dict:
    path = pubchem_dir / f"{paper_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_cid(q: CandidateQuestion, pubchem_data: dict) -> Optional[int]:
    """Return the first PubChem CID found in pubchem_links that matches a chemical entity."""
    resolved: list[dict] = pubchem_data.get("resolved", [])
    if not resolved:
        return None
    # Try to match by name against chemical_entities
    entities_lower = {e.lower() for e in q.chemical_entities}
    for entry in resolved:
        if entry.get("name", "").lower() in entities_lower:
            cid = entry.get("cid")
            if cid:
                return int(cid)
    # Fallback: return first resolved CID
    for entry in resolved:
        cid = entry.get("cid")
        if cid:
            return int(cid)
    return None


def _pubchem_hash(pubchem_data: dict) -> Optional[str]:
    """Return SHA-256 of the sorted CID list as a short provenance fingerprint."""
    resolved = pubchem_data.get("resolved", [])
    cids = sorted(str(e["cid"]) for e in resolved if e.get("cid"))
    if not cids:
        return None
    return hashlib.sha256(",".join(cids).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(
    selected_dir: Path,
    output_dir: Path,
    version: str,
    verification_path: Optional[Path] = None,
    critiques_dir: Optional[Path] = None,
    pubchem_dir: Optional[Path] = None,
) -> LiveChemBench:
    output_dir.mkdir(parents=True, exist_ok=True)
    verified_ids = _load_verified_ids(verification_path)

    critiques_dir = critiques_dir or (_REPO_ROOT / "data" / "critiques")
    pubchem_dir = pubchem_dir or (_REPO_ROOT / "data" / "pubchem_links")

    current_month = datetime.now(timezone.utc).strftime("%Y-%m")

    questions: list[BenchmarkQuestion] = []
    counter = 1
    dropped = 0

    for json_path in sorted(selected_dir.glob("*.json")):
        try:
            report = SelectedPaperQuestions.model_validate_json(json_path.read_text())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        raw_paper_id = report.paper_id
        formatted_paper_id = _format_paper_id(raw_paper_id)
        paper_source = _parse_source_type(raw_paper_id)

        critique_report = _load_critiques(critiques_dir, raw_paper_id)
        pubchem_data = _load_pubchem_links(pubchem_dir, raw_paper_id)
        pq_hash = _pubchem_hash(pubchem_data)

        passing = report.benchmark_ready()
        paper_added = 0

        for q in passing:
            q_id = f"{current_month}_{counter:03d}"
            verifier = _build_verifier(q)
            filters = _build_filters(q, critique_report)
            cid = _resolve_cid(q, pubchem_data)

            candidate = BenchmarkQuestion(
                id=q_id,
                paper_id=formatted_paper_id,
                segment_id=q.source_segment,
                cid=cid,
                question=q.question_text,
                answer=q.answer,
                answer_type=q.answer_type,
                answer_units=q.answer_units,
                tolerance=q.tolerance,
                question_type=q.question_type,
                chemical_entities=q.chemical_entities,
                verifier=verifier,
                filters=filters,
                provenance=Provenance(
                    month=current_month,
                    paper_source=paper_source,
                    conversion_tool="paddle_vl",
                    pubchem_query_log_hash=pq_hash,
                ),
            )
            counter += 1

            if verified_ids is not None and q_id not in verified_ids:
                logger.info("  dropping %s (failed verification)", q_id)
                dropped += 1
                continue

            questions.append(candidate)
            paper_added += 1

        logger.info("  %s → %d question(s) added", formatted_paper_id, paper_added)

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

    if dropped:
        logger.info("Dropped %d question(s) that failed verification.", dropped)
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
    parser.add_argument("--version", default="0.3.0", help="Benchmark version string.")
    parser.add_argument(
        "--selected-dir",
        default=str(_REPO_ROOT / "data" / "selected_questions"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "benchmark"),
    )
    parser.add_argument(
        "--critiques-dir",
        default=None,
        help="Path to critiques directory (default: data/critiques).",
    )
    parser.add_argument(
        "--pubchem-dir",
        default=None,
        help="Path to pubchem_links directory (default: data/pubchem_links).",
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
        critiques_dir=Path(args.critiques_dir) if args.critiques_dir else None,
        pubchem_dir=Path(args.pubchem_dir) if args.pubchem_dir else None,
    )


if __name__ == "__main__":
    main()
