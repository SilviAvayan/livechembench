"""
Question Repairer

For each paper:
  1. Loads proposed questions (data/proposed_questions/<id>.json)
  2. Loads critic reports (data/critiques/<id>.json)
  3. For questions where any critic flagged FAIL or NEEDS_REPAIR:
       - Calls an LLM to rewrite the question
       - Re-runs both critics on the repaired version
       - Keeps the repair if both critics now PASS, else drops it
  4. Questions where all critics already PASS are kept unchanged.
  5. Writes the full repair report to data/repaired_questions/<id>.json

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.question_repairer                    # all papers
    python -m src.agents.question_repairer --limit 3          # first 3 papers
    python -m src.agents.question_repairer --paper-id <id>    # single paper
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

from src.agents.critics import critique_question
from src.agents.models import (
    CandidateQuestion,
    CriticName,
    CriticVerdict,
    PaperCritiqueReport,
    PaperQuestions,
    RepairOutcome,
    RepairedPaperQuestions,
    RepairedQuestion,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _REPO_ROOT / "prompts"


def _load_prompt_config(name: str) -> dict:
    with open(_PROMPTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _needs_repair(critiques: PaperCritiqueReport, question_index: int) -> bool:
    """Return True if any critic did not PASS for this question."""
    for record in critiques.critiques:
        if record.question_index == question_index:
            if record.result.verdict != CriticVerdict.pass_:
                return True
    return False


def _build_repair_prompt(
    question: CandidateQuestion,
    critiques: PaperCritiqueReport,
    question_index: int,
) -> str:
    lines = ["ORIGINAL QUESTION:"]
    lines.append(json.dumps(question.model_dump(mode="json"), indent=2))
    lines.append("\nCRITIC FEEDBACK:")

    for record in critiques.critiques:
        if record.question_index != question_index:
            continue
        lines.append(f"\nCritic: {record.critic.value}")
        lines.append(f"Verdict: {record.result.verdict.value}")
        lines.append(f"Reason: {record.result.reason}")
        if record.result.suggested_fix:
            lines.append(f"Suggested fix: {record.result.suggested_fix}")

    lines.append("\nRewrite the question to fix all issues listed above.")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found: {text[:200]}")
    return json.loads(text[start:end])


def repair_question(
    question: CandidateQuestion,
    critiques: PaperCritiqueReport,
    question_index: int,
    client: OpenAI,
    cfg_repairer: dict,
    cfg_ill: dict,
    cfg_missing: dict,
) -> RepairedQuestion:
    """Attempt to repair a question and verify with critics."""
    user_msg = _build_repair_prompt(question, critiques, question_index)

    try:
        response = client.chat.completions.create(
            model=cfg_repairer["model"],
            max_tokens=cfg_repairer["max_tokens"],
            temperature=cfg_repairer["temperature"],
            messages=[
                {"role": "system", "content": cfg_repairer["system_prompt"]},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = _extract_json(response.choices[0].message.content)
        repaired_q = CandidateQuestion.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, Exception) as exc:
        logger.warning("Repair LLM call failed for Q%d: %s", question_index, exc)
        return RepairedQuestion(
            original=question,
            repaired=None,
            outcome=RepairOutcome.dropped,
            repair_notes=f"Repair LLM failed: {exc}",
        )

    # Re-run both critics on the repaired question
    verdicts: dict[CriticName, CriticVerdict] = {}
    for critic_name, cfg in [
        (CriticName.ill_defined, cfg_ill),
        (CriticName.missing_conditions, cfg_missing),
    ]:
        try:
            result = critique_question(repaired_q, critic_name, client, cfg)
            verdicts[critic_name] = result.verdict
            logger.info(
                "    re-critique Q%d [%s] → %s",
                question_index,
                critic_name.value,
                result.verdict.value,
            )
        except Exception as exc:
            logger.warning(
                "Re-critique failed for Q%d [%s]: %s", question_index, critic_name.value, exc
            )
            verdicts[critic_name] = CriticVerdict.fail

    all_pass = all(v == CriticVerdict.pass_ for v in verdicts.values())

    if all_pass:
        return RepairedQuestion(
            original=question,
            repaired=repaired_q,
            outcome=RepairOutcome.repaired,
            repair_notes="Repair verified: both critics now PASS.",
        )
    else:
        failing = [k.value for k, v in verdicts.items() if v != CriticVerdict.pass_]
        return RepairedQuestion(
            original=question,
            repaired=None,
            outcome=RepairOutcome.dropped,
            repair_notes=f"Repair failed re-critique: {', '.join(failing)} still not PASS.",
        )


def process_paper(
    paper_questions: PaperQuestions,
    critiques: PaperCritiqueReport,
    client: OpenAI,
    cfg_repairer: dict,
    cfg_ill: dict,
    cfg_missing: dict,
) -> RepairedPaperQuestions:
    repaired: list[RepairedQuestion] = []

    for idx, question in enumerate(paper_questions.questions):
        if not _needs_repair(critiques, idx):
            logger.info("  Q%d — all critics PASS, keeping as-is", idx)
            repaired.append(
                RepairedQuestion(
                    original=question,
                    repaired=None,
                    outcome=RepairOutcome.kept_original,
                    repair_notes=None,
                )
            )
        else:
            logger.info("  Q%d — needs repair, attempting...", idx)
            result = repair_question(
                question, critiques, idx, client, cfg_repairer, cfg_ill, cfg_missing
            )
            logger.info("  Q%d — outcome: %s", idx, result.outcome.value)
            repaired.append(result)

    return RepairedPaperQuestions(
        paper_id=paper_questions.paper_id,
        questions=repaired,
        repaired_at=datetime.now(timezone.utc).isoformat(),
    )


def run(
    proposed_dir: Path,
    critiques_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[RepairedPaperQuestions]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

    cfg_repairer = _load_prompt_config("question_repairer")
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

    results: list[RepairedPaperQuestions] = []

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("Proposed file not found, skipping: %s", json_path)
            continue

        critique_path = critiques_dir / json_path.name
        if not critique_path.exists():
            logger.warning("No critique file for %s — run critics first", json_path.stem)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already repaired, skipping: %s", json_path.stem)
            continue

        try:
            paper_questions = PaperQuestions.model_validate_json(json_path.read_text())
            critiques = PaperCritiqueReport.model_validate_json(critique_path.read_text())
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        if not paper_questions.questions:
            logger.info("No questions in %s, skipping", json_path.stem)
            continue

        logger.info(
            "Repairing %d question(s) for: %s",
            len(paper_questions.questions),
            paper_questions.paper_id,
        )

        report = process_paper(
            paper_questions, critiques, client, cfg_repairer, cfg_ill, cfg_missing
        )
        out_path.write_text(report.model_dump_json(indent=2))

        surviving = report.surviving()
        logger.info(
            "  → %d surviving question(s) for %s  "
            "(kept=%d  repaired=%d  dropped=%d)",
            len(surviving),
            report.paper_id,
            sum(1 for q in report.questions if q.outcome == RepairOutcome.kept_original),
            sum(1 for q in report.questions if q.outcome == RepairOutcome.repaired),
            sum(1 for q in report.questions if q.outcome == RepairOutcome.dropped),
        )
        results.append(report)

    logger.info("Done. %d paper(s) processed. Results in: %s", len(results), output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair flawed benchmark questions based on critic feedback."
    )
    parser.add_argument("--paper-id", default=None, help="Repair a single paper by ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process.")
    parser.add_argument(
        "--proposed-dir",
        default=str(_REPO_ROOT / "data" / "proposed_questions"),
    )
    parser.add_argument(
        "--critiques-dir",
        default=str(_REPO_ROOT / "data" / "critiques"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "repaired_questions"),
    )
    args = parser.parse_args()

    run(
        proposed_dir=Path(args.proposed_dir),
        critiques_dir=Path(args.critiques_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
