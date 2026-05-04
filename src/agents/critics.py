"""
Critics: Ill-defined and Missing-Conditions

Reads proposed question JSONs (data/proposed_questions/),
runs two critics on each question, and writes a critique report
to data/critiques/<paper_id>.json.

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.critics                    # all proposed papers
    python -m src.agents.critics --limit 3          # first 3 papers
    python -m src.agents.critics --paper-id <id>    # single paper
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
    CriticName,
    CriticResult,
    PaperCritiqueReport,
    PaperQuestions,
    QuestionCritiqueRecord,
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
    ]
    if question.answer_units:
        lines.append(f"answer_units: {question.answer_units}")
    lines.append(f"question_type: {question.question_type.value}")
    lines.append(f"verification_recipe: {question.verification_recipe}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


def critique_question(
    question: CandidateQuestion,
    critic_name: CriticName,
    client: OpenAI,
    cfg: dict,
) -> CriticResult:
    """Run a single critic on a single question."""
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
    return CriticResult.model_validate(raw)


def critique_paper(
    paper_questions: PaperQuestions,
    client: OpenAI,
    cfg_ill: dict,
    cfg_missing: dict,
) -> PaperCritiqueReport:
    """Run both critics on every question in a paper."""
    records: list[QuestionCritiqueRecord] = []
    now = datetime.now(timezone.utc).isoformat()

    for idx, question in enumerate(paper_questions.questions):
        for critic_name, cfg in [
            (CriticName.ill_defined, cfg_ill),
            (CriticName.missing_conditions, cfg_missing),
        ]:
            try:
                result = critique_question(question, critic_name, client, cfg)
            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                logger.warning(
                    "Critic %s failed for Q%d of %s: %s",
                    critic_name.value,
                    idx,
                    paper_questions.paper_id,
                    exc,
                )
                continue

            records.append(
                QuestionCritiqueRecord(
                    paper_id=paper_questions.paper_id,
                    question_index=idx,
                    question_text=question.question_text,
                    critic=critic_name,
                    result=result,
                    evaluated_at=now,
                )
            )
            logger.info(
                "  Q%d [%s] %s — %s",
                idx,
                critic_name.value,
                result.verdict.value,
                result.reason[:80],
            )

    return PaperCritiqueReport(
        paper_id=paper_questions.paper_id,
        critiques=records,
        critiqued_at=now,
    )


def run(
    proposed_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PaperCritiqueReport]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

    cfg_ill = _load_prompt_config("critic_ill_defined")
    cfg_missing = _load_prompt_config("critic_missing_conditions")

    client = OpenAI(
        api_key=api_key,
        base_url="https://inference-api.nvidia.com/v1",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    if paper_id:
        candidates = [proposed_dir / f"{paper_id}.json"]
    else:
        candidates = sorted(proposed_dir.glob("*.json"))
        if limit:
            candidates = candidates[:limit]

    if not candidates:
        logger.warning("No proposed question files found in %s", proposed_dir)

    results: list[PaperCritiqueReport] = []

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("File not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already critiqued, skipping: %s", json_path.stem)
            continue

        try:
            paper_questions = PaperQuestions.model_validate_json(json_path.read_text())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        if not paper_questions.questions:
            logger.info("No questions in %s, skipping", json_path.stem)
            continue

        logger.info(
            "Critiquing %d question(s) for: %s",
            len(paper_questions.questions),
            paper_questions.paper_id,
        )

        report = critique_paper(paper_questions, client, cfg_ill, cfg_missing)
        out_path.write_text(report.model_dump_json(indent=2))

        pass_count = sum(
            1 for c in report.critiques if c.result.verdict.value == "PASS"
        )
        logger.info(
            "  → %d critique(s) written (%d PASS) for %s",
            len(report.critiques),
            pass_count,
            report.paper_id,
        )
        results.append(report)

    logger.info("Done. %d paper(s) critiqued. Results in: %s", len(results), output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ill-defined and missing-conditions critics on proposed questions."
    )
    parser.add_argument("--paper-id", default=None, help="Critique a single paper by ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process.")
    parser.add_argument(
        "--proposed-dir",
        default=str(_REPO_ROOT / "data" / "proposed_questions"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "critiques"),
    )
    args = parser.parse_args()

    run(
        proposed_dir=Path(args.proposed_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
