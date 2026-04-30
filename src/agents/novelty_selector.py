"""Agent 1 — Segment Novelty Selector.

Reads all segmented papers and scores each one for scientific novelty.
Returns the top-K most novel papers along with the best segment text to
use as the source for question generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import BaseAgent

log = logging.getLogger(__name__)

_SYSTEM = """\
You are a chemistry research analyst. Your job is to evaluate the scientific \
novelty of a chemistry paper segment and return a structured JSON response.

Novelty criteria (score 0.0–1.0):
  1.0  Breakthrough: new reaction, new material class, new mechanism, new method
  0.7  Significant: meaningful extension of prior work with clear new finding
  0.4  Incremental: applies known techniques to new substrate / system
  0.1  Routine: replication, commentary, metadata, or non-chemistry content

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "novelty_score": <float 0.0–1.0>,
  "novelty_reason": "<one sentence>",
  "best_segment": "<abstract|key_points|conclusion>",
  "best_segment_text": "<the actual text of the chosen segment>"
}
"""

_USER_TMPL = """\
Paper ID: {paper_id}
Title: {title}

=== Abstract ===
{abstract}

=== Key Points ===
{key_points}

=== Conclusion ===
{conclusion}

Score the novelty of this paper and select the single most informative segment \
(abstract, key_points, or conclusion) to use for question generation.
"""


@dataclass
class NoveltyResult:
    paper_id: str
    title: str
    novelty_score: float
    novelty_reason: str
    best_segment: str                 # "abstract" | "key_points" | "conclusion"
    best_segment_text: str
    raw_paper: dict[str, Any] = field(default_factory=dict, repr=False)


class NoveltySelector(BaseAgent):
    """Scores segmented papers for novelty and selects the best segment."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        top_k: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def score_paper(self, paper: dict[str, Any]) -> NoveltyResult:
        """Score a single segmented paper."""
        key_points_text = "\n".join(
            f"  • {kp}" for kp in paper.get("key_points", [])
        )
        user_msg = _USER_TMPL.format(
            paper_id=paper.get("paper_id", "unknown"),
            title=paper.get("title", "(no title)"),
            abstract=paper.get("abstract", "")[:2000],
            key_points=key_points_text[:2000],
            conclusion=paper.get("conclusion", "")[:1500],
        )
        try:
            result = await self.chat_json(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # Resolve best_segment_text if the model omitted it
            seg = result.get("best_segment", "abstract")
            seg_text = result.get("best_segment_text") or _resolve_segment(paper, seg)

            return NoveltyResult(
                paper_id=paper["paper_id"],
                title=paper.get("title", ""),
                novelty_score=float(result.get("novelty_score", 0.0)),
                novelty_reason=result.get("novelty_reason", ""),
                best_segment=seg,
                best_segment_text=seg_text,
                raw_paper=paper,
            )
        except Exception as exc:
            log.warning("Novelty scoring failed for %s: %s", paper.get("paper_id"), exc)
            return NoveltyResult(
                paper_id=paper.get("paper_id", "unknown"),
                title=paper.get("title", ""),
                novelty_score=0.0,
                novelty_reason=f"Scoring error: {exc}",
                best_segment="abstract",
                best_segment_text=paper.get("abstract", ""),
                raw_paper=paper,
            )

    async def select(self, segmented_dir: Path) -> list[NoveltyResult]:
        """Score all papers in *segmented_dir* and return the top-K.

        Skips the _summary.json file.
        """
        paper_files = [
            p for p in segmented_dir.glob("*.json")
            if p.name != "_summary.json"
        ]
        if not paper_files:
            log.warning("No segmented papers found in %s", segmented_dir)
            return []

        log.info("Scoring %d papers for novelty …", len(paper_files))

        # Load all papers
        papers: list[dict[str, Any]] = []
        for pf in paper_files:
            try:
                papers.append(json.loads(pf.read_text()))
            except Exception as exc:
                log.warning("Could not load %s: %s", pf.name, exc)

        # Skip papers with no real content
        papers = [
            p for p in papers
            if p.get("extraction_status") != "failed"
            and (p.get("abstract") or p.get("conclusion") or p.get("key_points"))
        ]

        # Score concurrently (one coroutine per paper)
        results = await asyncio.gather(
            *[self.score_paper(p) for p in papers], return_exceptions=True
        )

        valid: list[NoveltyResult] = []
        for r in results:
            if isinstance(r, Exception):
                log.warning("Skipping paper due to error: %s", r)
            else:
                valid.append(r)

        # Sort descending and return top-K
        valid.sort(key=lambda x: x.novelty_score, reverse=True)
        chosen = valid[: self.top_k]

        log.info(
            "Selected %d/%d papers (top novelty scores: %s)",
            len(chosen),
            len(valid),
            [f"{r.novelty_score:.2f}" for r in chosen],
        )
        return chosen


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_segment(paper: dict[str, Any], seg: str) -> str:
    if seg == "key_points":
        return "\n".join(paper.get("key_points", []))
    if seg == "conclusion":
        return paper.get("conclusion", "")
    return paper.get("abstract", "")
