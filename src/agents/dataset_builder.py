"""Agent 5 — Dataset Builder.

Converts accepted VerifiedQuestion objects into three output artefacts:

  dataset.jsonl        — one line per question, in the LiveChemBench record format
  provenance.jsonl     — full audit trail (segment text, critic logs, repair count)
  verifier_configs.json — machine-readable configs for automated answer verification
"""

from __future__ import annotations

import json
from typing import Union
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .tri_critic_verifier import VerifiedQuestion

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output record schemas
# ---------------------------------------------------------------------------

def _dataset_record(vq: VerifiedQuestion) -> dict:
    q = vq.question
    return {
        "id": q.id,
        "paper_id": q.paper_id,
        "question_type": q.question_type,
        "question": q.question,
        "choices": q.choices,
        "answer_key": q.answer_key,
        "answer": q.answer,
        "difficulty": q.difficulty,
        "subject": q.subject,
        "entities": q.entities,
        "segment_type": q.segment_type,
        "source_quote": q.source_quote,
    }


def _provenance_record(vq: VerifiedQuestion, novelty_scores: dict[str, float]) -> dict:
    q = vq.question
    return {
        "id": q.id,
        "paper_id": q.paper_id,
        "novelty_score": novelty_scores.get(q.paper_id, 0.0),
        "repair_count": vq.repair_count,
        "final_verdict": vq.final_verdict,
        "source_text_excerpt": q.source_text[:500],
        "verification_plan": q.verification_plan,
        "critics_log": [
            [
                {
                    "critic": v.critic,
                    "verdict": v.verdict,
                    "reason": v.reason,
                    "repaired_question": v.repaired_question,
                }
                for v in iteration
            ]
            for iteration in vq.critics_log
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _verifier_config(vq: VerifiedQuestion) -> dict:
    q = vq.question
    config: dict = {
        "id": q.id,
        "question_type": q.question_type,
        "verification_plan": q.verification_plan,
        "source_quote": q.source_quote,
    }
    if q.question_type == "multiple_choice":
        config["strategy"] = "exact_match_key"
        config["correct_key"] = q.answer_key
        config["correct_text"] = q.answer
    elif q.question_type == "numerical":
        config["strategy"] = "numeric_tolerance"
        config["expected"] = q.answer
        config["tolerance_pct"] = 5.0
    else:
        config["strategy"] = "llm_judge"
        config["reference_answer"] = q.answer
        config["judge_prompt"] = (
            "Does the model answer convey the same meaning as the reference answer? "
            "Reply with a JSON object: {\"correct\": true/false, \"reason\": \"...\"}."
        )
    return config


# ---------------------------------------------------------------------------
# Builder class
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Writes the three output JSONL/JSON artefacts from verified questions."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        provenance_path: Union[str, Path],
        verifier_configs_path: Union[str, Path],
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.provenance_path = Path(provenance_path)
        self.verifier_configs_path = Path(verifier_configs_path)

    def build(
        self,
        accepted: list[VerifiedQuestion],
        novelty_scores: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Write all three output files. Returns record counts."""
        if novelty_scores is None:
            novelty_scores = {}

        # Ensure parent dirs exist
        for p in (self.dataset_path, self.provenance_path, self.verifier_configs_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        dataset_records = [_dataset_record(vq) for vq in accepted]
        provenance_records = [_provenance_record(vq, novelty_scores) for vq in accepted]
        verifier_configs = {vq.question.id: _verifier_config(vq) for vq in accepted}

        # Write dataset JSONL
        with self.dataset_path.open("w", encoding="utf-8") as f:
            for rec in dataset_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write provenance JSONL
        with self.provenance_path.open("w", encoding="utf-8") as f:
            for rec in provenance_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # Write verifier configs JSON
        with self.verifier_configs_path.open("w", encoding="utf-8") as f:
            json.dump(verifier_configs, f, indent=2, ensure_ascii=False)

        counts = {
            "dataset_records": len(dataset_records),
            "provenance_records": len(provenance_records),
            "verifier_configs": len(verifier_configs),
        }
        log.info(
            "Dataset built: %d questions → %s",
            len(dataset_records),
            self.dataset_path,
        )
        return counts
