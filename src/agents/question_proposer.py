"""Agent 3 — Question Proposer.

Given a paper segment and its linked chemical entities, the agent drafts
candidate chemistry questions of three types:
  • multiple_choice  — 4-option MCQ with a single correct answer
  • free_text        — open-ended short-answer question
  • numerical        — requires a numeric calculation or lookup answer

Each candidate also carries a *verification_plan* describing how to check
the answer against the source text.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent
from .entity_extractor import ChemicalEntity
from .novelty_selector import NoveltyResult

log = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert chemistry exam question writer for a benchmark dataset.
Your goal is to generate high-quality, unambiguous questions that test \
genuine chemistry knowledge derived from a research paper segment.

For each question return the following JSON structure:

For multiple_choice:
{
  "question_type": "multiple_choice",
  "question": "<question text>",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_key": "<A|B|C|D>",
  "answer": "<full text of the correct choice>",
  "difficulty": "<easy|medium|hard>",
  "subject": "<reaction chemistry|materials|spectroscopy|biochemistry|computational|other>",
  "source_quote": "<verbatim quote from the segment that supports the answer>",
  "verification_plan": "<how to verify the answer from the text>"
}

For free_text:
{
  "question_type": "free_text",
  "question": "<question text>",
  "choices": {},
  "answer_key": "",
  "answer": "<expected answer>",
  "difficulty": "<easy|medium|hard>",
  "subject": "...",
  "source_quote": "...",
  "verification_plan": "..."
}

For numerical:
{
  "question_type": "numerical",
  "question": "<question text>",
  "choices": {},
  "answer_key": "",
  "answer": "<numeric value with units>",
  "difficulty": "<easy|medium|hard>",
  "subject": "...",
  "source_quote": "...",
  "verification_plan": "..."
}

Return a JSON object with a single key "questions" whose value is a list of \
the above question objects. Generate exactly {n_questions} questions covering \
a mix of the requested types: {types}.

Rules:
- Every question must be answerable solely from the provided segment.
- MCQ distractors must be plausible but clearly wrong upon careful reading.
- Numerical questions must include explicit units.
- Never repeat the exact wording of the question in the answer.
- source_quote must be a verbatim excerpt (≤ 150 chars) from the segment.
"""

_USER_TMPL = """\
Paper ID: {paper_id}
Segment type: {segment_type}

=== Source Segment ===
{segment_text}

=== Linked Chemical Entities ===
{entities_block}

Generate {n_questions} questions now.
"""


@dataclass
class CandidateQuestion:
    id: str
    paper_id: str
    segment_type: str
    question_type: str                     # multiple_choice | free_text | numerical
    question: str
    choices: dict[str, str]                # {"A": ..., "B": ..., ...}  empty for non-MCQ
    answer_key: str                        # "A" | "B" | ... | ""
    answer: str
    difficulty: str
    subject: str
    source_quote: str
    verification_plan: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    source_text: str = field(default="", repr=False)


class QuestionProposer(BaseAgent):
    """Proposes candidate questions from a segment + entity context."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        questions_per_paper: int = 5,
        question_types: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.questions_per_paper = questions_per_paper
        self.question_types = question_types or [
            "multiple_choice",
            "free_text",
            "numerical",
        ]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _entity_block(self, entities: list[ChemicalEntity]) -> str:
        if not entities:
            return "(none identified)"
        lines = []
        for e in entities:
            parts = [f"• {e.mention} ({e.entity_type})"]
            if e.cid:
                parts.append(f"  CID={e.cid}")
            if e.canonical_smiles:
                parts.append(f"  SMILES={e.canonical_smiles}")
            if e.molecular_formula:
                parts.append(f"  Formula={e.molecular_formula}")
            lines.append("\n".join(parts))
        return "\n".join(lines)

    async def propose(
        self,
        novelty_result: NoveltyResult,
        entities: list[ChemicalEntity],
    ) -> list[CandidateQuestion]:
        """Draft candidate questions for a single paper's segment."""
        types_str = ", ".join(self.question_types)
        system = _SYSTEM.format(
            n_questions=self.questions_per_paper,
            types=types_str,
        )
        user = _USER_TMPL.format(
            paper_id=novelty_result.paper_id,
            segment_type=novelty_result.best_segment,
            segment_text=novelty_result.best_segment_text[:4000],
            entities_block=self._entity_block(entities),
            n_questions=self.questions_per_paper,
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
            raw_questions: list[dict[str, Any]] = result.get("questions", [])
        except Exception as exc:
            log.warning("Question proposal failed for %s: %s", novelty_result.paper_id, exc)
            return []

        candidates: list[CandidateQuestion] = []
        entity_dicts = [
            {
                "mention": e.mention,
                "cid": e.cid,
                "smiles": e.canonical_smiles,
                "formula": e.molecular_formula,
                "pubchem_url": e.pubchem_url,
            }
            for e in entities
            if e.cid is not None
        ]

        for q in raw_questions:
            if not q.get("question"):
                continue
            candidates.append(
                CandidateQuestion(
                    id=f"q_{uuid.uuid4().hex[:8]}",
                    paper_id=novelty_result.paper_id,
                    segment_type=novelty_result.best_segment,
                    question_type=q.get("question_type", "free_text"),
                    question=q.get("question", ""),
                    choices=q.get("choices") or {},
                    answer_key=q.get("answer_key", ""),
                    answer=q.get("answer", ""),
                    difficulty=q.get("difficulty", "medium"),
                    subject=q.get("subject", "other"),
                    source_quote=q.get("source_quote", ""),
                    verification_plan=q.get("verification_plan", ""),
                    entities=entity_dicts,
                    source_text=novelty_result.best_segment_text,
                )
            )
        log.info(
            "Proposed %d questions for %s", len(candidates), novelty_result.paper_id
        )
        return candidates

    async def run(
        self,
        novelty_results: list[NoveltyResult],
        entity_map: dict[str, list[ChemicalEntity]],
    ) -> list[CandidateQuestion]:
        """Propose questions for all selected papers concurrently."""
        tasks = [
            self.propose(nr, entity_map.get(nr.paper_id, []))
            for nr in novelty_results
        ]
        per_paper = await asyncio.gather(*tasks, return_exceptions=True)
        all_questions: list[CandidateQuestion] = []
        for item in per_paper:
            if isinstance(item, Exception):
                log.warning("Proposal error: %s", item)
            else:
                all_questions.extend(item)
        log.info("Total candidate questions: %d", len(all_questions))
        return all_questions
