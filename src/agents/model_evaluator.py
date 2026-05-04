"""
Model Evaluator

Prompts an LLM with each question in the benchmark (no answer provided),
parses its response, scores it against the ground truth, and writes an
evaluation report.

Required environment variable:
    NVIDIA_API_KEY — your NVIDIA inference API key

Usage:
    python -m src.agents.model_evaluator
    python -m src.agents.model_evaluator --benchmark data/benchmark/livechembench_v0.1.0.json
    python -m src.agents.model_evaluator --model gcp/google/gemini-2.0-flash
    python -m src.agents.model_evaluator --question-id lcb_0001
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from openai import OpenAI

from src.agents.models import (
    AnswerType,
    BenchmarkQuestion,
    EvalReport,
    EvalResult,
    EvalScores,
    LiveChemBench,
    QuestionType,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PROMPT_FILE = _REPO_ROOT / "prompts" / "model_evaluator.yaml"


def _load_prompt_config(model_override: Optional[str] = None) -> dict:
    with open(_PROMPT_FILE) as f:
        cfg = yaml.safe_load(f)
    if model_override:
        cfg["model"] = model_override
    return cfg


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer(
    response: str,
    answer_type: AnswerType,
    chemical_entities: list[str],
) -> str:
    text = response.strip()

    if answer_type == AnswerType.float_:
        m = re.search(r"[-+]?\d*\.?\d+", text)
        return m.group() if m else text

    if answer_type == AnswerType.int_:
        m = re.search(r"\d+", text)
        return m.group() if m else text

    if answer_type == AnswerType.choice:
        # Check if any known chemical entity appears verbatim in the response
        text_lower = text.lower()
        for entity in chemical_entities:
            if entity.lower() in text_lower:
                return entity
        # Fall back to the full response (first line only)
        return text.split("\n")[0].strip()

    # string: return first line, stripped
    return text.split("\n")[0].strip()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(extracted: str, expected: str, q: BenchmarkQuestion) -> bool:
    e_str = extracted.strip()
    x_str = expected.strip()

    if q.answer_type == AnswerType.float_:
        try:
            tol = q.tolerance if q.tolerance is not None else 0.01
            return abs(float(e_str) - float(x_str)) <= tol
        except ValueError:
            return False

    if q.answer_type == AnswerType.int_:
        try:
            return int(e_str) == int(x_str)
        except ValueError:
            return e_str.lower() == x_str.lower()

    # string / choice — blank response never counts as correct
    if not e_str:
        return False
    if e_str.lower() == x_str.lower():
        return True
    # Accept if expected value appears inside extracted text (e.g. "The answer is UDMA")
    # but NOT the reverse — a short extracted answer appearing inside a long expected
    # string would be a false positive (e.g. '' in 'UDMA').
    return x_str.lower() in e_str.lower()


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    benchmark: LiveChemBench,
    client: OpenAI,
    cfg: dict,
    question_id: Optional[str] = None,
) -> EvalReport:
    questions = benchmark.questions
    if question_id:
        questions = [q for q in questions if q.id == question_id]

    results: list[EvalResult] = []

    for q in questions:
        logger.info("Evaluating %s [%s] ...", q.id, q.question_type.value)
        try:
            response = client.chat.completions.create(
                model=cfg["model"],
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                messages=[
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": q.question},
                ],
            )
            msg = response.choices[0].message
            raw = msg.content
            if raw is None:
                # Some models (e.g. reasoning/MoE models) put the answer in a
                # non-standard field or return structured output.  Try known
                # alternate fields before falling back to empty.
                raw = getattr(msg, "reasoning_content", None) or ""
                if raw:
                    logger.info("  %s content=None; fell back to reasoning_content", q.id)
                else:
                    logger.warning(
                        "  %s content=None, finish_reason=%s — recording as blank",
                        q.id, response.choices[0].finish_reason,
                    )
            raw = raw.strip()
        except Exception as exc:
            logger.error("  %s API error: %s", q.id, exc)
            raw = ""

        extracted = _extract_answer(raw, q.answer_type, q.chemical_entities)
        correct = _score(extracted, q.answer, q)

        logger.info(
            "  %s | expected=%r  model=%r  correct=%s",
            q.id, q.answer, extracted, correct,
        )

        results.append(EvalResult(
            question_id=q.id,
            paper_id=q.paper_id,
            question_type=q.question_type,
            answer_type=q.answer_type,
            question_text=q.question,
            expected_answer=q.answer,
            model_raw_response=raw,
            model_answer=extracted,
            correct=correct,
        ))

    scores = _compute_scores(results)
    logger.info(
        "Evaluation complete: %.1f%% correct (%d/%d)",
        scores.overall * 100, scores.n_correct, scores.n_total,
    )
    _log_breakdown(scores)

    return EvalReport(
        model=cfg["model"],
        benchmark_version=benchmark.version,
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        scores=scores,
        results=results,
    )


def _compute_scores(results: list[EvalResult]) -> EvalScores:
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_paper: dict[str, list[bool]] = defaultdict(list)
    by_answer_type: dict[str, list[bool]] = defaultdict(list)

    for r in results:
        by_type[r.question_type.value].append(r.correct)
        by_paper[r.paper_id].append(r.correct)
        by_answer_type[r.answer_type.value].append(r.correct)

    def _acc(lst: list[bool]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    n_correct = sum(1 for r in results if r.correct)
    n_total = len(results)

    return EvalScores(
        overall=_acc([r.correct for r in results]),
        by_type={k: _acc(v) for k, v in by_type.items()},
        by_paper={k: _acc(v) for k, v in by_paper.items()},
        by_answer_type={k: _acc(v) for k, v in by_answer_type.items()},
        n_correct=n_correct,
        n_total=n_total,
    )


def _log_breakdown(scores: EvalScores) -> None:
    logger.info("=" * 50)
    logger.info("Overall accuracy : %.1f%% (%d/%d)",
                scores.overall * 100, scores.n_correct, scores.n_total)
    logger.info("By question type : %s",
                {k: f"{v*100:.0f}%" for k, v in scores.by_type.items()})
    logger.info("By answer type   : %s",
                {k: f"{v*100:.0f}%" for k, v in scores.by_answer_type.items()})
    logger.info("By paper         : %s",
                {k: f"{v*100:.0f}%" for k, v in scores.by_paper.items()})
    logger.info("=" * 50)


def run(
    benchmark_path: Path,
    output_dir: Path,
    model_override: Optional[str] = None,
    question_id: Optional[str] = None,
    base_url: str = "https://inference-api.nvidia.com/v1",
    api_key_env: str = "NVIDIA_API_KEY",
) -> EvalReport:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"{api_key_env} environment variable is not set.")

    cfg = _load_prompt_config(model_override)
    client = OpenAI(api_key=api_key, base_url=base_url)

    benchmark = LiveChemBench.model_validate_json(benchmark_path.read_text())
    report = evaluate(benchmark, client, cfg, question_id=question_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = cfg["model"].replace("/", "_").replace(":", "_")
    out_path = output_dir / f"v{benchmark.version}_{model_slug}.json"
    out_path.write_text(report.model_dump_json(indent=2))
    logger.info("Eval report written to: %s", out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM against the LiveChemBench benchmark."
    )
    parser.add_argument(
        "--benchmark",
        default=str(_REPO_ROOT / "data" / "benchmark" / "livechembench_v0.3.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "eval_results"),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to evaluate, e.g. 'nvidia/openai/gpt-oss-120b'.",
    )
    parser.add_argument(
        "--base-url",
        default="https://inference-api.nvidia.com/v1",
        help="OpenAI-compatible API base URL (default: NVIDIA Inference API).",
    )
    parser.add_argument(
        "--api-key-env",
        default="NVIDIA_API_KEY",
        help="Name of the environment variable holding the API key (default: NVIDIA_API_KEY).",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Evaluate a single question by ID (e.g. 2026-05_001).",
    )
    args = parser.parse_args()

    run(
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        model_override=args.model,
        question_id=args.question_id,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
    )


if __name__ == "__main__":
    main()
