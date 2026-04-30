"""Agent 4 — Tri-Critic Verifier Loop.

Three independent LLM critics each evaluate a candidate question:

  1. Factuality Critic  — Can the answer be confirmed from the source text?
  2. Clarity Critic     — Is the question unambiguous and well-formed?
  3. Chemistry Critic   — Is the chemistry accurate and does it test real knowledge?

Each critic returns one of:
  ACCEPT  — question passes this criterion
  REPAIR  — question has fixable issues (critic provides repaired version)
  REJECT  — question is fundamentally flawed and should be discarded

The loop runs up to max_iterations rounds. A question is accepted into the
final dataset only when all three critics return ACCEPT in the same round.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

from .base import BaseAgent
from .question_proposer import CandidateQuestion

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critic prompt templates
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_TMPL = """\
You are the {role}.

Your job: evaluate the candidate question below against the provided source \
segment and return a structured JSON verdict.

Return ONLY valid JSON (no markdown fences):
{{
  "verdict": "<ACCEPT|REPAIR|REJECT>",
  "reason": "<one or two sentences>",
  "repaired_question": "<full corrected question text, or null if ACCEPT/REJECT>",
  "repaired_answer": "<corrected answer, or null if ACCEPT/REJECT>",
  "repaired_choices": {{}} // corrected MCQ choices dict, or {{}} if unchanged
}}

Evaluation criteria for {role}:
{criteria}
"""

_CRITIC_USER_TMPL = """\
=== Source Segment ===
{source_text}

=== Candidate Question ===
Type: {question_type}
Question: {question}
{choices_block}
Answer: {answer}
Source quote: {source_quote}
Verification plan: {verification_plan}

Evaluate now.
"""

_CRITICS = {
    "Factuality Critic": (
        "- The answer must be directly supported by the source segment.\n"
        "- The source_quote must be a verbatim substring of the source text.\n"
        "- If the answer cannot be confirmed without external knowledge, REJECT.\n"
        "- If the quote is wrong but the answer is otherwise verifiable, REPAIR."
    ),
    "Clarity Critic": (
        "- The question must be unambiguous — a careful reader cannot interpret it\n"
        "  two ways.\n"
        "- For MCQ, exactly one choice must be correct; distractors must be plausible\n"
        "  but wrong.\n"
        "- Grammatical errors, undefined abbreviations, or missing units → REPAIR.\n"
        "- If the question is inherently ambiguous and cannot be fixed → REJECT."
    ),
    "Chemistry Critic": (
        "- Chemical names, formulas, SMILES, and properties must be correct.\n"
        "- Numerical answers must include correct units.\n"
        "- The question must test genuine chemistry knowledge, not trivial recall.\n"
        "- Incorrect chemistry that can be fixed → REPAIR; fundamentally wrong → REJECT."
    ),
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CriticVerdict:
    critic: str
    verdict: str         # ACCEPT | REPAIR | REJECT
    reason: str
    repaired_question: Optional[str] = None
    repaired_answer: Optional[str] = None
    repaired_choices: dict[str, str] = field(default_factory=dict)


@dataclass
class VerifiedQuestion:
    question: CandidateQuestion
    critics_log: list[list[CriticVerdict]]   # one list per iteration
    repair_count: int
    final_verdict: str                       # ACCEPT | REJECT


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TriCriticVerifier(BaseAgent):
    """Runs three critics iteratively until consensus or exhaustion."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        max_iterations: int = 3,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------

    def _choices_block(self, q: CandidateQuestion) -> str:
        if not q.choices:
            return ""
        lines = ["Choices:"]
        for k, v in q.choices.items():
            marker = " ← correct" if k == q.answer_key else ""
            lines.append(f"  {k}) {v}{marker}")
        return "\n".join(lines)

    async def _call_critic(
        self, critic_name: str, criteria: str, q: CandidateQuestion
    ) -> CriticVerdict:
        system = _CRITIC_SYSTEM_TMPL.format(
            role=critic_name, criteria=criteria
        )
        user = _CRITIC_USER_TMPL.format(
            source_text=q.source_text[:3000],
            question_type=q.question_type,
            question=q.question,
            choices_block=self._choices_block(q),
            answer=q.answer,
            source_quote=q.source_quote,
            verification_plan=q.verification_plan,
        )
        try:
            result = await self.chat_json(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            verdict = result.get("verdict", "REJECT").upper()
            if verdict not in ("ACCEPT", "REPAIR", "REJECT"):
                verdict = "REJECT"
            return CriticVerdict(
                critic=critic_name,
                verdict=verdict,
                reason=result.get("reason", ""),
                repaired_question=result.get("repaired_question"),
                repaired_answer=result.get("repaired_answer"),
                repaired_choices=result.get("repaired_choices") or {},
            )
        except Exception as exc:
            log.warning("Critic '%s' failed: %s", critic_name, exc)
            return CriticVerdict(
                critic=critic_name,
                verdict="REJECT",
                reason=f"Critic error: {exc}",
            )

    def _apply_repairs(
        self, q: CandidateQuestion, verdicts: list[CriticVerdict]
    ) -> tuple[CandidateQuestion, int]:
        """Merge all REPAIR suggestions into a new CandidateQuestion."""
        repaired = copy.deepcopy(q)
        repairs_applied = 0
        for v in verdicts:
            if v.verdict == "REPAIR":
                if v.repaired_question:
                    repaired.question = v.repaired_question
                    repairs_applied += 1
                if v.repaired_answer:
                    repaired.answer = v.repaired_answer
                    repairs_applied += 1
                if v.repaired_choices:
                    repaired.choices = v.repaired_choices
                    repairs_applied += 1
        return repaired, repairs_applied

    async def verify_question(
        self, q: CandidateQuestion
    ) -> VerifiedQuestion:
        """Run the tri-critic loop for a single question."""
        current = copy.deepcopy(q)
        all_iterations: list[list[CriticVerdict]] = []
        total_repairs = 0

        for iteration in range(self.max_iterations):
            # Run all three critics concurrently
            verdicts = await asyncio.gather(
                *[
                    self._call_critic(name, criteria, current)
                    for name, criteria in _CRITICS.items()
                ]
            )
            verdicts = list(verdicts)
            all_iterations.append(verdicts)

            accept_count = sum(1 for v in verdicts if v.verdict == "ACCEPT")
            reject_count = sum(1 for v in verdicts if v.verdict == "REJECT")

            log.debug(
                "  iter=%d q=%s accepts=%d rejects=%d",
                iteration + 1, current.id, accept_count, reject_count,
            )

            if reject_count > 0:
                # Any rejection is final
                return VerifiedQuestion(
                    question=current,
                    critics_log=all_iterations,
                    repair_count=total_repairs,
                    final_verdict="REJECT",
                )

            if accept_count == 3:
                return VerifiedQuestion(
                    question=current,
                    critics_log=all_iterations,
                    repair_count=total_repairs,
                    final_verdict="ACCEPT",
                )

            # Some repairs requested — apply and loop
            current, n = self._apply_repairs(current, verdicts)
            total_repairs += n

        # Exhausted iterations without full consensus
        return VerifiedQuestion(
            question=current,
            critics_log=all_iterations,
            repair_count=total_repairs,
            final_verdict="REJECT",
        )

    async def run(
        self, candidates: list[CandidateQuestion]
    ) -> tuple[list[VerifiedQuestion], list[VerifiedQuestion]]:
        """Verify all candidates; return (accepted, rejected)."""
        log.info("Running tri-critic verification on %d candidates …", len(candidates))
        results = await asyncio.gather(
            *[self.verify_question(q) for q in candidates], return_exceptions=True
        )
        accepted: list[VerifiedQuestion] = []
        rejected: list[VerifiedQuestion] = []
        for item in results:
            if isinstance(item, Exception):
                log.warning("Verification error: %s", item)
                continue
            if item.final_verdict == "ACCEPT":
                accepted.append(item)
            else:
                rejected.append(item)

        log.info(
            "Verification complete: %d accepted, %d rejected",
            len(accepted),
            len(rejected),
        )
        return accepted, rejected
