"""
Paper Quality Evaluator Agent

Reads segmented paper JSONs, calls the NVIDIA-hosted LLM via the OpenAI-compatible
API, and returns a validated PaperQualityEvaluation Pydantic object.

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.paper_quality_evaluator                  # all papers
    python -m src.agents.paper_quality_evaluator --limit 5        # first 5 papers
    python -m src.agents.paper_quality_evaluator --paper-id <id>  # single paper
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

from src.agents.models import PaperQualityEvaluation
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_PROMPT_FILE = _PROMPTS_DIR / "paper_quality_evaluator.yaml"


def _load_prompt_config() -> dict:
    with open(_PROMPT_FILE) as f:
        return yaml.safe_load(f)


def _build_user_message(paper: dict) -> str:
    compression_ratio = (
        round(paper["compressed_char_count"] / paper["raw_char_count"] * 100, 1)
        if paper.get("raw_char_count")
        else "N/A"
    )
    abstract_preview = (paper.get("abstract") or "")[:400].strip()
    key_points_count = len(paper.get("key_points") or [])
    tables_count = len(paper.get("tables") or [])
    figures_count = len(paper.get("figure_paths") or [])

    return f"""Evaluate the following segmented document.

--- DOCUMENT METADATA ---
paper_id        : {paper.get("paper_id")}
source_file     : {paper.get("source_file")}
title           : {paper.get("title") or "(none)"}
extraction_status: {paper.get("extraction_status")}
engine          : {paper.get("engine")}

--- EXTRACTION STATISTICS ---
section_count         : {paper.get("section_count", 0)}
raw_char_count        : {paper.get("raw_char_count", 0)}
compressed_char_count : {paper.get("compressed_char_count", 0)}
compression_ratio     : {compression_ratio}%
key_points_extracted  : {key_points_count}
tables_extracted      : {tables_count}
figures_extracted     : {figures_count}

--- ABSTRACT (first 400 chars) ---
{abstract_preview or "(empty)"}

--- CONCLUSION ---
{(paper.get("conclusion") or "(empty)")[:300].strip()}

Respond with a single JSON object and nothing else. Use exactly these keys:
{{
  "document_type": one of "research_paper" | "review" | "supplementary" | "dataset" | "protocol" | "other",
  "is_real_paper": true or false,
  "ocr_quality": one of "good" | "partial" | "poor",
  "has_abstract": true or false,
  "has_figures": true or false,
  "has_tables": true or false,
  "worth_pursuing": true or false,
  "justification": "natural language explanation"
}}"""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    return json.loads(text[start:end])


def evaluate_paper(
    paper: dict,
    client: OpenAI,
    cfg: dict,
) -> PaperQualityEvaluation:
    """Evaluate a single segmented paper and return a validated Pydantic object."""
    response = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": _build_user_message(paper)},
        ],
    )

    raw_args = _extract_json(response.choices[0].message.content)

    # Inject fields the model doesn't need to generate
    raw_args["paper_id"] = paper["paper_id"]
    raw_args["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    return PaperQualityEvaluation.model_validate(raw_args)


def run(
    segmented_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PaperQualityEvaluation]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable is not set.")

    cfg = _load_prompt_config()

    client = OpenAI(
        api_key=api_key,
        base_url="https://inference-api.nvidia.com/v1",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect JSON files to evaluate
    if paper_id:
        candidates = [segmented_dir / f"{paper_id}.json"]
    else:
        candidates = sorted(
            p for p in segmented_dir.glob("*.json") if p.stem != "_summary"
        )
        if limit:
            candidates = candidates[:limit]

    results: list[PaperQualityEvaluation] = []

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("File not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already evaluated, skipping: %s", json_path.stem)
            continue

        with open(json_path) as f:
            paper = json.load(f)

        logger.info("Evaluating: %s", paper.get("paper_id"))
        try:
            evaluation = evaluate_paper(paper, client, cfg)
        except (ValidationError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Evaluation failed for %s: %s", json_path.stem, exc)
            continue

        with open(out_path, "w") as f:
            f.write(evaluation.model_dump_json(indent=2))

        logger.info(
            "  → %s | worth_pursuing=%s | ocr=%s | type=%s",
            evaluation.paper_id,
            evaluation.worth_pursuing,
            evaluation.ocr_quality.value,
            evaluation.document_type.value,
        )
        results.append(evaluation)

    logger.info("Evaluated %d paper(s). Results in: %s", len(results), output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate segmented paper quality using an LLM."
    )
    parser.add_argument("--paper-id", default=None, help="Evaluate a single paper by ID.")
    parser.add_argument("--limit", type=int, default=None, help="Max papers to evaluate.")
    parser.add_argument(
        "--segmented-dir",
        default=str(_REPO_ROOT / "data" / "segmented_papers"),
        help="Directory containing segmented JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "evaluations"),
        help="Directory to write evaluation JSON files.",
    )
    args = parser.parse_args()

    run(
        segmented_dir=Path(args.segmented_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
