"""
Question Proposer Agent

Reads segmented paper JSONs that passed quality evaluation (worth_pursuing=True),
calls the NVIDIA-hosted LLM, and returns validated CandidateQuestion Pydantic objects.

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.question_proposer                  # all worthy papers
    python -m src.agents.question_proposer --limit 3        # first 3 papers
    python -m src.agents.question_proposer --paper-id <id>  # single paper
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

from src.agents.models import CandidateQuestion, PaperQuestions
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPT_FILE = _REPO_ROOT / "prompts" / "question_proposer.yaml"


def _load_prompt_config() -> dict:
    with open(_PROMPT_FILE) as f:
        return yaml.safe_load(f)


def _build_user_message(paper: dict, n_questions: int) -> str:
    abstract = (paper.get("abstract") or "")[:1000].strip()
    conclusion = (paper.get("conclusion") or "")[:500].strip()
    key_points = paper.get("key_points") or []
    tables = paper.get("tables") or []

    key_points_text = "\n".join(f"- {kp[:300]}" for kp in key_points[:5])
    tables_text = "\n\n".join(t[:500] for t in tables[:3])

    return f"""Generate up to {n_questions} benchmark questions from this chemistry paper.

--- PAPER ---
paper_id : {paper.get("paper_id")}
title    : {paper.get("title") or "(none)"}

--- ABSTRACT ---
{abstract or "(empty)"}

--- KEY POINTS ---
{key_points_text or "(none)"}

--- CONCLUSION ---
{conclusion or "(empty)"}

--- TABLES (first 3) ---
{tables_text or "(none)"}

Return a JSON array of question objects as specified. If insufficient chemistry content, return []."""


def _extract_json_array(text: str) -> list:
    """Extract the first JSON array from a model response."""
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    return json.loads(text[start:end])


def propose_questions(
    paper: dict,
    client: OpenAI,
    cfg: dict,
) -> PaperQuestions:
    """Generate candidate questions for a single paper."""
    n = cfg.get("questions_per_paper", 5)

    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": _build_user_message(paper, n)},
        ],
    )

    raw = _extract_json_array(response.choices[0].message.content)

    questions = []
    for item in raw:
        try:
            questions.append(CandidateQuestion.model_validate(item))
        except ValidationError as exc:
            logger.warning("Skipping invalid question: %s", exc)

    return PaperQuestions(
        paper_id=paper["paper_id"],
        questions=questions,
        proposed_at=datetime.now(timezone.utc).isoformat(),
    )


def run(
    segmented_dir: Path,
    evaluations_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PaperQuestions]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

    cfg = _load_prompt_config()

    client = OpenAI(
        api_key=api_key,
        base_url="https://inference-api.nvidia.com/v1",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Only process papers marked worth_pursuing=True by the evaluator
    if paper_id:
        candidates = [segmented_dir / f"{paper_id}.json"]
    else:
        eval_files = sorted(evaluations_dir.glob("*.json"))
        worthy_ids = set()
        for ef in eval_files:
            try:
                ev = json.loads(ef.read_text())
                if ev.get("worth_pursuing"):
                    worthy_ids.add(ev["paper_id"])
            except (json.JSONDecodeError, KeyError):
                continue

        candidates = sorted(
            segmented_dir / f"{pid}.json" for pid in worthy_ids
            if (segmented_dir / f"{pid}.json").exists()
        )
        if limit:
            candidates = candidates[:limit]

    if not candidates:
        logger.warning(
            "No worthy papers found. Run paper_quality_evaluator first, "
            "or pass --paper-id to force a specific paper."
        )

    results: list[PaperQuestions] = []

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("Segmented file not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already proposed, skipping: %s", json_path.stem)
            continue

        paper = json.loads(json_path.read_text())
        logger.info("Proposing questions for: %s", paper.get("paper_id"))

        try:
            result = propose_questions(paper, client, cfg)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("Failed for %s: %s", json_path.stem, exc)
            continue

        out_path.write_text(result.model_dump_json(indent=2))
        logger.info(
            "  → %d question(s) proposed for %s",
            len(result.questions),
            result.paper_id,
        )
        results.append(result)

    logger.info("Done. %d paper(s) processed. Results in: %s", len(results), output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate candidate benchmark questions from segmented chemistry papers."
    )
    parser.add_argument("--paper-id", default=None, help="Propose for a single paper by ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to process.")
    parser.add_argument(
        "--segmented-dir",
        default=str(_REPO_ROOT / "data" / "segmented_papers"),
    )
    parser.add_argument(
        "--evaluations-dir",
        default=str(_REPO_ROOT / "data" / "evaluations"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "proposed_questions"),
    )
    args = parser.parse_args()

    run(
        segmented_dir=Path(args.segmented_dir),
        evaluations_dir=Path(args.evaluations_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
