"""
A1 — Segment Selector (Novelty-Driven Segment Bandit)

For each worthy paper, scores and ranks its segments using a UCB-like policy that
balances novelty prior N(s) with empirical acceptance reward R(s).

Novelty prior N(s) is computed from two signals:
  1. Rhetorical role: results/conclusion segments are preferred over background.
  2. Chemical entity density: segments mentioning more unique chemical names score higher.

Empirical reward R(s) tracks how often questions from this segment TYPE have been
accepted (survived critics + novelty check). Loaded from data/segment_rewards.json
and updated after each pipeline run.

Outputs: data/segment_selections/<paper_id>.json
  - ranked list of segments with scores and selection rationale

Usage:
    python -m src.agents.segment_selector                   # all worthy papers
    python -m src.agents.segment_selector --paper-id <id>  # single paper
    python -m src.agents.segment_selector --top-k 3        # keep top 3 segments
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REWARDS_FILE = _REPO_ROOT / "data" / "segment_rewards.json"

# Rhetorical role weights: how novel/information-rich each segment type is
_ROLE_WEIGHTS: dict[str, float] = {
    "results":      1.0,
    "conclusion":   0.95,
    "discussion":   0.85,
    "methods":      0.6,
    "introduction": 0.3,
    "background":   0.2,
    "abstract":     0.5,   # high density but often duplicated elsewhere
    "key_points":   0.7,
    "tables":       0.9,   # often contain quantitative results
    "figures":      0.4,
    "other":        0.3,
}

# Chemical entity patterns (used to compute entity density)
_ENTITY_PATTERNS = [
    r"\b[A-Z]{2,}[\'\-]?\d{2,}[a-zA-Z]?\b",   # compound codes: GYY4137, NSC-12345
    r"\b[A-Z][a-z]+[0-9][a-zA-Z0-9\-]*\b",      # CamelCase+digits: Compound1a
    r"\([A-Z][a-zA-Z0-9\-\' ]{3,30}\)",          # parenthetical names
    r"\b(?:IC50|EC50|Kd|Ki|GI50|LD50)\b",        # bioactivity markers
    r"\bSMILES\b",                                # explicit SMILES mention
    r"\bC\d{1,2}H\d{1,2}",                        # molecular formula fragment
]


def _rhetorical_role(title: str) -> str:
    """Infer a segment's rhetorical role from its title (lowercase match)."""
    t = (title or "").lower()
    for role in _ROLE_WEIGHTS:
        if role in t:
            return role
    # Fallback heuristics
    if any(k in t for k in ("result", "finding", "outcome")):
        return "results"
    if any(k in t for k in ("conclu", "summary", "remark")):
        return "conclusion"
    if any(k in t for k in ("discuss", "interpretation")):
        return "discussion"
    if any(k in t for k in ("method", "material", "procedure", "protocol", "experi")):
        return "methods"
    if any(k in t for k in ("intro", "background", "related", "prior", "literature")):
        return "introduction"
    if any(k in t for k in ("table", "figure", "fig.", "chart")):
        return "tables"
    return "other"


def _entity_density(text: str) -> float:
    """
    Fraction of words that match chemical entity patterns.
    Returns a value in [0, 1].
    """
    if not text:
        return 0.0
    n_words = max(1, len(text.split()))
    hits = sum(len(re.findall(p, text)) for p in _ENTITY_PATTERNS)
    return min(1.0, hits / n_words * 10)  # scale so ~1 hit per 10 words → 1.0


def _novelty_prior(segment: dict) -> float:
    """
    N(s) ∈ [0, 1]: rhetorical weight × entity density × length bonus.
    """
    title = segment.get("title") or segment.get("type") or ""
    text = segment.get("content") or segment.get("text") or ""

    role = _rhetorical_role(title)
    rw = _ROLE_WEIGHTS.get(role, 0.3)
    ed = _entity_density(text)
    # Length bonus: longer segments (up to ~2000 chars) tend to have more content
    length_bonus = min(1.0, len(text) / 2000)

    return rw * (0.5 + 0.3 * ed + 0.2 * length_bonus)


def _load_rewards() -> dict[str, dict]:
    """
    Load empirical rewards per segment role.
    Structure: {"results": {"accepted": 5, "attempted": 8}, ...}
    """
    if _REWARDS_FILE.exists():
        try:
            return json.loads(_REWARDS_FILE.read_text())
        except Exception:
            pass
    return {}


def _empirical_reward(role: str, rewards: dict) -> float:
    """
    R(s) ∈ [0, 1]: fraction of questions from this role that were accepted.
    Returns 0.5 (neutral) if no history for this role.
    """
    r = rewards.get(role, {})
    attempted = r.get("attempted", 0)
    accepted = r.get("accepted", 0)
    if attempted == 0:
        return 0.5  # neutral prior with no history
    return accepted / attempted


def _ucb_score(novelty: float, reward: float, n_total: int, n_role: int, c: float = 1.0) -> float:
    """
    UCB1-style score combining novelty prior, empirical reward, and exploration bonus.

    score = 0.6 * N(s) + 0.4 * R(s) + c * sqrt(ln(n_total + 1) / (n_role + 1))
    """
    exploration = c * math.sqrt(math.log(n_total + 1) / (n_role + 1))
    return 0.6 * novelty + 0.4 * reward + exploration


def score_segments(paper: dict, rewards: dict, top_k: Optional[int] = None) -> list[dict]:
    """
    Score and rank all segments in a paper. Returns sorted list (highest first).
    """
    paper_id = paper.get("paper_id", "unknown")
    segments = []

    # Collect all content-bearing segments
    raw_segments: list[dict] = []

    # Abstract as a pseudo-segment
    if paper.get("abstract"):
        raw_segments.append({"title": "abstract", "content": paper["abstract"], "type": "abstract"})

    # Key points
    kp = paper.get("key_points")
    if kp:
        raw_segments.append({"title": "key_points", "content": " ".join(kp), "type": "key_points"})

    # Named sections
    for section in paper.get("sections") or []:
        raw_segments.append({
            "title": section.get("title") or section.get("heading") or "",
            "content": section.get("content") or section.get("text") or "",
            "type": "section",
        })

    # Tables — entries may be dicts with "rows" or plain strings
    for i, table in enumerate(paper.get("tables") or []):
        if isinstance(table, str):
            table_text = table
        elif isinstance(table, dict):
            table_text = " ".join(
                " ".join(str(cell) for cell in row)
                for row in (table.get("rows") or [])
            )
        else:
            table_text = str(table)
        raw_segments.append({
            "title": f"table_{i}",
            "content": table_text,
            "type": "tables",
        })

    # Conclusion
    if paper.get("conclusion"):
        raw_segments.append({"title": "conclusion", "content": paper["conclusion"], "type": "conclusion"})

    # Count how many times each role has been selected so far (for UCB exploration)
    n_total = sum(rewards.get(r, {}).get("attempted", 0) for r in _ROLE_WEIGHTS)
    role_counts = {r: rewards.get(r, {}).get("attempted", 0) for r in _ROLE_WEIGHTS}

    for seg in raw_segments:
        if not (seg.get("content") or "").strip():
            continue
        role = _rhetorical_role(seg["title"])
        n_prior = novelty_prior = _novelty_prior(seg)
        r = _empirical_reward(role, rewards)
        n_role = role_counts.get(role, 0)
        score = _ucb_score(n_prior, r, n_total, n_role)

        segments.append({
            "title": seg["title"],
            "role": role,
            "content_preview": (seg.get("content") or "")[:200],
            "novelty_prior": round(n_prior, 4),
            "empirical_reward": round(r, 4),
            "ucb_score": round(score, 4),
            "selected": False,
        })

    # Sort descending by UCB score
    segments.sort(key=lambda x: x["ucb_score"], reverse=True)

    # Mark top-k as selected
    if top_k:
        for seg in segments[:top_k]:
            seg["selected"] = True
    else:
        for seg in segments:
            seg["selected"] = True

    logger.info(
        "  %s: %d segments scored; top=%s (score=%.3f)",
        paper_id,
        len(segments),
        segments[0]["title"] if segments else "—",
        segments[0]["ucb_score"] if segments else 0,
    )
    return segments


def update_rewards(segment_role: str, accepted: bool) -> None:
    """
    Update the empirical reward for a segment role after a question attempt.
    Call this from the question_proposer or critic pipeline.
    """
    rewards = _load_rewards()
    if segment_role not in rewards:
        rewards[segment_role] = {"accepted": 0, "attempted": 0}
    rewards[segment_role]["attempted"] += 1
    if accepted:
        rewards[segment_role]["accepted"] += 1
    _REWARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REWARDS_FILE.write_text(json.dumps(rewards, indent=2))


def run(
    segmented_dir: Path,
    evaluations_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    top_k: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rewards = _load_rewards()

    worthy_ids: set[str] = set()
    if paper_id:
        worthy_ids.add(paper_id.removesuffix(".json"))
    else:
        for ev_path in evaluations_dir.glob("*.json"):
            try:
                ev = json.loads(ev_path.read_text())
                if ev.get("worth_pursuing"):
                    worthy_ids.add(ev["paper_id"])
            except Exception:
                pass

    candidates = (
        [segmented_dir / f"{paper_id}.json"]
        if paper_id
        else sorted(
            p for p in segmented_dir.glob("*.json")
            if p.stem in worthy_ids and not p.stem.startswith("_")
        )
    )
    if limit:
        candidates = candidates[:limit]

    logger.info("Segment Selector: processing %d paper(s)", len(candidates))

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("File not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already selected, skipping: %s", json_path.stem)
            continue

        try:
            paper = json.loads(json_path.read_text())
        except Exception as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        logger.info("Scoring segments: %s", json_path.stem)
        scored = score_segments(paper, rewards, top_k=top_k)

        result = {
            "paper_id": paper.get("paper_id", json_path.stem),
            "segments": scored,
            "top_k": top_k,
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path.write_text(json.dumps(result, indent=2))

    logger.info("Done. Results in: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1: Score and select segments for question generation (novelty bandit)."
    )
    parser.add_argument("--paper-id", default=None, help="Score segments for a single paper.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top segments to mark as selected (default: 3).",
    )
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
        default=str(_REPO_ROOT / "data" / "segment_selections"),
    )
    args = parser.parse_args()

    run(
        segmented_dir=Path(args.segmented_dir),
        evaluations_dir=Path(args.evaluations_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        top_k=args.top_k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
