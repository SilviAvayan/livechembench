"""
Novelty Selector

Reads repaired question JSONs (data/repaired_questions/), runs the novelty critic
on each surviving question, and writes benchmark-ready questions to
data/selected_questions/<paper_id>.json.

Only questions that PASS the novelty check are included in the benchmark.

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.novelty_selector                    # all papers
    python -m src.agents.novelty_selector --limit 3          # first 3 papers
    python -m src.agents.novelty_selector --paper-id <id>    # single paper
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI
from pydantic import ValidationError

from src.agents.models import (
    CandidateQuestion,
    NoveltyResult,
    NoveltyVerdict,
    RepairedPaperQuestions,
    SelectedPaperQuestions,
    SelectedQuestion,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _REPO_ROOT / "prompts"


def _load_prompt_config(name: str) -> dict:
    with open(_PROMPTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _build_user_message(question: CandidateQuestion) -> str:
    lines = [
        f"question_text: {question.question_text}",
        f"answer: {question.answer}",
        f"answer_type: {question.answer_type.value}",
        f"question_type: {question.question_type.value}",
        f"chemical_entities: {', '.join(question.chemical_entities)}",
        f"verification_recipe: {question.verification_recipe}",
    ]
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(text[start:end])


def check_novelty(
    question: CandidateQuestion,
    client: OpenAI,
    cfg: dict,
) -> NoveltyResult:
    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": _build_user_message(question)},
        ],
    )
    raw = _extract_json(response.choices[0].message.content)
    # Novelty critic only uses PASS/FAIL — coerce suggested_fix if present
    raw.pop("suggested_fix", None)
    return NoveltyResult.model_validate(raw)


def select_paper(
    repaired: RepairedPaperQuestions,
    client: OpenAI,
    cfg: dict,
) -> SelectedPaperQuestions:
    surviving = repaired.surviving()
    selected: list[SelectedQuestion] = []

    for question in surviving:
        try:
            result = check_novelty(question, client, cfg)
        except (json.JSONDecodeError, ValidationError, Exception) as exc:
            logger.warning(
                "Novelty check failed for Q in %s: %s — defaulting to FAIL",
                repaired.paper_id,
                exc,
            )
            result = NoveltyResult(
                verdict=NoveltyVerdict.fail,
                reason=f"Novelty check error: {exc}",
            )

        logger.info(
            "  [novelty] %s — %s",
            result.verdict.value,
            result.reason[:80],
        )
        selected.append(
            SelectedQuestion(
                question=question,
                novelty_verdict=result.verdict,
                novelty_reason=result.reason,
            )
        )

    return SelectedPaperQuestions(
        paper_id=repaired.paper_id,
        questions=selected,
        selected_at=datetime.now(timezone.utc).isoformat(),
    )


def run(
    repaired_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[SelectedPaperQuestions]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

    cfg = _load_prompt_config("critic_novelty")

    client = OpenAI(
        api_key=api_key,
        base_url="https://inference-api.nvidia.com/v1",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if paper_id:
        candidates = [repaired_dir / f"{paper_id}.json"]
    else:
        candidates = sorted(repaired_dir.glob("*.json"))
        if limit:
            candidates = candidates[:limit]

    if not candidates:
        logger.warning("No repaired question files found in %s", repaired_dir)

    results: list[SelectedPaperQuestions] = []

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("File not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already selected, skipping: %s", json_path.stem)
            continue

        try:
            repaired = RepairedPaperQuestions.model_validate_json(json_path.read_text())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        surviving = repaired.surviving()
        if not surviving:
            logger.info("No surviving questions in %s, skipping", json_path.stem)
            continue

        logger.info(
            "Running novelty check on %d question(s) for: %s",
            len(surviving),
            repaired.paper_id,
        )

        report = select_paper(repaired, client, cfg)
        out_path.write_text(report.model_dump_json(indent=2))

        benchmark_ready = report.benchmark_ready()
        logger.info(
            "  → %d / %d question(s) passed novelty for %s",
            len(benchmark_ready),
            len(surviving),
            report.paper_id,
        )
        results.append(report)

    logger.info("Done. %d paper(s) processed. Results in: %s", len(results), output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select benchmark-ready questions by novelty check."
    )
    parser.add_argument("--paper-id", default=None, help="Select for a single paper by ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process.")
    parser.add_argument(
        "--repaired-dir",
        default=str(_REPO_ROOT / "data" / "repaired_questions"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "selected_questions"),
    )
    args = parser.parse_args()

    run(
        repaired_dir=Path(args.repaired_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
