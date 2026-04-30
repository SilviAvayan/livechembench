"""Agent 6 — Evaluator.

Runs each question in dataset.jsonl through one or more baseline models and
scores the responses using the verifier_configs strategies:

  • exact_match_key   — MCQ: model answer is the letter (A/B/C/D)
  • numeric_tolerance — numerical: float comparison with tolerance %
  • llm_judge        — free text: ask an LLM judge to compare

Produces:
  leaderboard.json  — per-model accuracy + per-difficulty/type breakdown
  leaderboard.md    — human-readable markdown table
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from .base import BaseAgent

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_exact_key(model_answer: str, correct_key: str) -> bool:
    """Extract A/B/C/D letter from model response and compare."""
    # Look for standalone A/B/C/D at start of response or after "Answer:"
    clean = model_answer.strip().upper()
    # Common patterns: "A", "A.", "(A)", "Answer: A", "The answer is A"
    m = re.search(r"\b([A-D])\b", clean)
    if m:
        return m.group(1) == correct_key.upper()
    return False


def _score_numeric(model_answer: str, expected: str, tolerance_pct: float) -> bool:
    """Extract a number from the model response and compare within tolerance."""
    def extract_float(s: str) -> Optional[float]:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group()) if m else None

    got = extract_float(model_answer)
    ref = extract_float(expected)
    if got is None or ref is None or ref == 0:
        return False
    return abs(got - ref) / abs(ref) * 100 <= tolerance_pct


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    model_name: str
    question_id: str
    question_type: str
    difficulty: str
    subject: str
    model_answer: str
    is_correct: bool
    judge_reason: str = ""


@dataclass
class LeaderboardEntry:
    model_name: str
    total: int
    correct: int
    accuracy: float
    by_type: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)
    by_subject: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

_MCQ_SYSTEM = (
    "You are taking a chemistry exam. Read the question and options carefully, "
    "then respond with ONLY the letter of the correct answer (A, B, C, or D). "
    "Nothing else."
)

_FREE_SYSTEM = (
    "You are a chemistry expert. Answer the question concisely and precisely. "
    "For numerical answers, include the value and units."
)

_JUDGE_SYSTEM = (
    "You are an answer judge. Compare the model answer to the reference answer and "
    "decide if they convey the same correct information.\n"
    "Return ONLY valid JSON: {\"correct\": true/false, \"reason\": \"...\"}"
)


class Evaluator:
    """Runs baseline models against the dataset and scores them."""

    def __init__(
        self,
        api_key: str,
        primary_model: str,
        primary_base_url: str,
        baseline_models: Optional[List[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.api_key = api_key
        self.primary_model = primary_model
        self.primary_base_url = primary_base_url
        self.baseline_models = baseline_models or []
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _make_client(self, base_url: str) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    def _build_prompt(self, record: dict) -> list[dict]:
        qt = record["question_type"]
        q = record["question"]
        choices = record.get("choices", {})

        if qt == "multiple_choice" and choices:
            opts = "\n".join(f"{k}) {v}" for k, v in choices.items())
            content = f"{q}\n\nOptions:\n{opts}"
            system = _MCQ_SYSTEM
        else:
            content = q
            system = _FREE_SYSTEM

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    async def _judge_free_text(
        self,
        judge_client: AsyncOpenAI,
        judge_model: str,
        model_answer: str,
        reference: str,
        judge_prompt: str,
    ) -> tuple[bool, str]:
        user = (
            f"Reference answer: {reference}\n\n"
            f"Model answer: {model_answer}\n\n"
            f"{judge_prompt}"
        )
        try:
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
            from .base import _parse_json
            parsed = _parse_json(raw)
            return bool(parsed.get("correct")), parsed.get("reason", "")
        except Exception as exc:
            log.debug("Judge failed: %s", exc)
            return False, str(exc)

    async def _evaluate_one(
        self,
        client: AsyncOpenAI,
        model_name: str,
        record: dict,
        verifier: dict,
        judge_client: AsyncOpenAI,
        judge_model: str,
    ) -> ModelResult:
        messages = self._build_prompt(record)
        try:
            resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            model_answer = resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("Model call failed for %s on %s: %s", model_name, record["id"], exc)
            model_answer = ""

        strategy = verifier.get("strategy", "llm_judge")
        is_correct = False
        judge_reason = ""

        if strategy == "exact_match_key":
            is_correct = _score_exact_key(model_answer, verifier.get("correct_key", ""))
        elif strategy == "numeric_tolerance":
            is_correct = _score_numeric(
                model_answer,
                verifier.get("expected", ""),
                verifier.get("tolerance_pct", 5.0),
            )
        else:  # llm_judge
            is_correct, judge_reason = await self._judge_free_text(
                judge_client,
                judge_model,
                model_answer,
                verifier.get("reference_answer", ""),
                verifier.get("judge_prompt", ""),
            )

        return ModelResult(
            model_name=model_name,
            question_id=record["id"],
            question_type=record["question_type"],
            difficulty=record.get("difficulty", "medium"),
            subject=record.get("subject", "other"),
            model_answer=model_answer,
            is_correct=is_correct,
            judge_reason=judge_reason,
        )

    async def _run_model(
        self,
        model_name: str,
        base_url: str,
        dataset: list[dict],
        verifiers: dict[str, dict],
    ) -> list[ModelResult]:
        client = self._make_client(base_url)
        judge_client = self._make_client(self.primary_base_url)

        log.info("Evaluating model: %s (%d questions) …", model_name, len(dataset))
        results = await asyncio.gather(
            *[
                self._evaluate_one(
                    client,
                    model_name,
                    rec,
                    verifiers.get(rec["id"], {}),
                    judge_client,
                    self.primary_model,
                )
                for rec in dataset
            ],
            return_exceptions=True,
        )
        valid = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("Evaluation error: %s", r)
            else:
                valid.append(r)
        return valid

    def _compute_leaderboard(self, all_results: list[list[ModelResult]]) -> list[LeaderboardEntry]:
        entries: list[LeaderboardEntry] = []
        for model_results in all_results:
            if not model_results:
                continue
            model_name = model_results[0].model_name

            def _breakdown(attr: str) -> dict:
                groups: dict[str, list[bool]] = {}
                for r in model_results:
                    key = getattr(r, attr)
                    groups.setdefault(key, []).append(r.is_correct)
                return {
                    k: {"correct": sum(v), "total": len(v), "accuracy": sum(v) / len(v)}
                    for k, v in groups.items()
                }

            total = len(model_results)
            correct = sum(r.is_correct for r in model_results)
            entries.append(
                LeaderboardEntry(
                    model_name=model_name,
                    total=total,
                    correct=correct,
                    accuracy=correct / total if total else 0.0,
                    by_type=_breakdown("question_type"),
                    by_difficulty=_breakdown("difficulty"),
                    by_subject=_breakdown("subject"),
                )
            )
        entries.sort(key=lambda e: e.accuracy, reverse=True)
        return entries

    def _write_leaderboard(
        self,
        entries: list[LeaderboardEntry],
        json_path: Path,
        md_path: Path,
    ) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": [
                {
                    "model_name": e.model_name,
                    "total": e.total,
                    "correct": e.correct,
                    "accuracy": round(e.accuracy * 100, 1),
                    "by_type": e.by_type,
                    "by_difficulty": e.by_difficulty,
                    "by_subject": e.by_subject,
                }
                for e in entries
            ],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        # Markdown
        lines = [
            "# LiveChemBench Leaderboard",
            f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n",
            "| Rank | Model | Accuracy | Correct / Total |",
            "|------|-------|----------|-----------------|",
        ]
        for rank, e in enumerate(entries, 1):
            lines.append(
                f"| {rank} | {e.model_name} | {e.accuracy*100:.1f}% | {e.correct}/{e.total} |"
            )

        # Per-type breakdown for top model
        if entries:
            top = entries[0]
            lines += [
                f"\n## {top.model_name} — breakdown by question type\n",
                "| Type | Correct | Total | Accuracy |",
                "|------|---------|-------|----------|",
            ]
            for qt, stats in top.by_type.items():
                lines.append(
                    f"| {qt} | {stats['correct']} | {stats['total']} | {stats['accuracy']*100:.1f}% |"
                )

        md_path.write_text("\n".join(lines) + "\n")
        log.info("Leaderboard written to %s and %s", json_path, md_path)

    async def run(
        self,
        dataset_path: Path,
        verifier_configs_path: Path,
        leaderboard_json: Path,
        leaderboard_md: Path,
    ) -> None:
        """Full evaluation run: load dataset → score models → write leaderboard."""
        # Load dataset
        records = [
            json.loads(line)
            for line in dataset_path.read_text().splitlines()
            if line.strip()
        ]
        verifiers: dict[str, dict] = json.loads(verifier_configs_path.read_text())

        if not records:
            log.warning("Dataset is empty — nothing to evaluate.")
            return

        # Collect all models to evaluate
        models_to_eval = [(self.primary_model, self.primary_base_url)] + [
            (m["name"], m["base_url"]) for m in self.baseline_models
        ]
        # Deduplicate
        seen: set[str] = set()
        unique_models = []
        for name, url in models_to_eval:
            if name not in seen:
                seen.add(name)
                unique_models.append((name, url))

        # Run all models (sequentially to avoid rate limiting)
        all_results: list[list[ModelResult]] = []
        for name, url in unique_models:
            results = await self._run_model(name, url, records, verifiers)
            all_results.append(results)

        entries = self._compute_leaderboard(all_results)
        self._write_leaderboard(entries, leaderboard_json, leaderboard_md)
