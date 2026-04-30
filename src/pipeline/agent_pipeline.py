"""AgentPipeline — orchestrates all six agents end-to-end.

Usage (async):
    pipeline = AgentPipeline(config)
    await pipeline.run()

Usage (CLI / sync):
    pipeline = AgentPipeline(config)
    pipeline.run_sync()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.config.models import AppConfig
from src.agents.novelty_selector import NoveltySelector
from src.agents.entity_extractor import EntityExtractor
from src.agents.question_proposer import QuestionProposer
from src.agents.tri_critic_verifier import TriCriticVerifier
from src.agents.dataset_builder import DatasetBuilder
from src.agents.evaluator import Evaluator

log = logging.getLogger(__name__)


class AgentPipeline:
    """Wires together all six agents using the application config."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        ac = cfg.agents                        # AgentConfig
        out = cfg.output                       # OutputConfig

        api_key = ac.api_key
        model = ac.primary_model
        base_url = ac.base_url

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Export it as an environment variable or set agents.api_key in config.yaml."
            )

        self.novelty_selector = NoveltySelector(
            api_key=api_key,
            model=model,
            base_url=base_url,
            top_k=ac.novelty_selector.top_k,
            temperature=ac.novelty_selector.temperature,
            max_tokens=ac.novelty_selector.max_tokens,
        )
        self.entity_extractor = EntityExtractor(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=ac.entity_extractor.temperature,
            max_tokens=ac.entity_extractor.max_tokens,
            pubchem_timeout=ac.entity_extractor.pubchem_timeout,
            max_entities_per_paper=ac.entity_extractor.max_entities_per_paper,
        )
        self.question_proposer = QuestionProposer(
            api_key=api_key,
            model=model,
            base_url=base_url,
            questions_per_paper=ac.question_proposer.questions_per_paper,
            question_types=ac.question_proposer.question_types,
            temperature=ac.question_proposer.temperature,
            max_tokens=ac.question_proposer.max_tokens,
        )
        self.tri_critic = TriCriticVerifier(
            api_key=api_key,
            model=model,
            base_url=base_url,
            max_iterations=ac.tri_critic.max_iterations,
            temperature=ac.tri_critic.temperature,
            max_tokens=ac.tri_critic.max_tokens,
        )
        self.dataset_builder = DatasetBuilder(
            dataset_path=out.dataset_jsonl,
            provenance_path=out.provenance_jsonl,
            verifier_configs_path=out.verifier_configs,
        )
        self.evaluator = Evaluator(
            api_key=api_key,
            primary_model=model,
            primary_base_url=base_url,
            baseline_models=[
                {"name": m.name, "base_url": m.base_url}
                for m in ac.evaluator.baseline_models
            ],
            temperature=ac.evaluator.temperature,
            max_tokens=ac.evaluator.max_tokens,
        )
        self._out = out

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    async def run(self, skip_eval: bool = False) -> None:
        """Run the full pipeline from novelty selection through evaluation."""
        segmented_dir = Path(self.cfg.paths.segmented_papers)

        # ── Agent 1: Novelty Selection ────────────────────────────────
        log.info("=== Agent 1: Novelty Selector ===")
        novelty_results = await self.novelty_selector.select(segmented_dir)
        if not novelty_results:
            log.error("No papers with valid content found. Aborting.")
            return

        # ── Agent 2: Entity Extraction + PubChem ─────────────────────
        log.info("=== Agent 2: Entity Extractor + PubChem Linker ===")
        entity_map = await self.entity_extractor.run(novelty_results)

        # ── Agent 3: Question Proposal ────────────────────────────────
        log.info("=== Agent 3: Question Proposer ===")
        candidates = await self.question_proposer.run(novelty_results, entity_map)
        if not candidates:
            log.error("No candidate questions generated. Aborting.")
            return

        # ── Agent 4: Tri-Critic Verification ─────────────────────────
        log.info("=== Agent 4: Tri-Critic Verifier ===")
        accepted, rejected = await self.tri_critic.run(candidates)
        log.info(
            "Verification: %d accepted / %d rejected (%.0f%% acceptance rate)",
            len(accepted),
            len(rejected),
            len(accepted) / max(len(candidates), 1) * 100,
        )
        if not accepted:
            log.error("All questions were rejected by the critics. Aborting.")
            return

        # ── Agent 5: Dataset Builder ──────────────────────────────────
        log.info("=== Agent 5: Dataset Builder ===")
        novelty_scores = {nr.paper_id: nr.novelty_score for nr in novelty_results}
        counts = self.dataset_builder.build(accepted, novelty_scores)
        log.info("Dataset written: %s", counts)

        # ── Agent 6: Evaluator ────────────────────────────────────────
        if skip_eval:
            log.info("Skipping evaluation (skip_eval=True).")
            return

        log.info("=== Agent 6: Evaluator ===")
        await self.evaluator.run(
            dataset_path=Path(self._out.dataset_jsonl),
            verifier_configs_path=Path(self._out.verifier_configs),
            leaderboard_json=Path(self._out.leaderboard_json),
            leaderboard_md=Path(self._out.leaderboard_md),
        )
        log.info("Pipeline complete. Leaderboard: %s", self._out.leaderboard_md)

    def run_sync(self, skip_eval: bool = False) -> None:
        """Synchronous entry point (wraps asyncio.run)."""
        asyncio.run(self.run(skip_eval=skip_eval))
