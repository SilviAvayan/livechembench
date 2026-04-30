"""Agent 2 — Entity Extractor + PubChem Linker.

1. Uses the LLM to extract chemical entity mentions from a text segment.
2. For each entity, queries the PubChem REST API to resolve a CID and fetch
   canonical properties (IUPAC name, molecular formula, InChIKey, SMILES).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import aiohttp

from .base import BaseAgent
from .novelty_selector import NoveltyResult

log = logging.getLogger(__name__)

_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

_SYSTEM = """\
You are a chemistry NLP expert. Extract all named chemical entities \
(compounds, reagents, catalysts, solvents, polymers, biomolecules) \
from the supplied text.

Return ONLY valid JSON — no markdown fences:
{
  "entities": [
    {
      "mention": "<exact string from text>",
      "normalized_name": "<preferred IUPAC or common name>",
      "entity_type": "<compound|reagent|catalyst|solvent|polymer|biomolecule|other>"
    }
  ]
}

Rules:
- Include only specific named chemicals (not vague terms like "compound" or "material").
- De-duplicate by normalized_name.
- Limit to at most {max_entities} entities.
"""

_USER_TMPL = """\
Text:
{text}

Extract all chemical entity mentions.
"""


@dataclass
class ChemicalEntity:
    mention: str
    normalized_name: str
    entity_type: str
    cid: Optional[int] = None
    iupac_name: str = ""
    molecular_formula: str = ""
    inchikey: str = ""
    canonical_smiles: str = ""
    pubchem_url: str = ""


class EntityExtractor(BaseAgent):
    """Extracts chemical entities from text and links them to PubChem."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        pubchem_timeout: int = 10,
        max_entities_per_paper: int = 20,
    ) -> None:
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pubchem_timeout = aiohttp.ClientTimeout(total=pubchem_timeout)
        self.max_entities = max_entities_per_paper

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def extract_entities(self, text: str) -> list[ChemicalEntity]:
        """Return raw (un-linked) entities from *text*."""
        system = _SYSTEM.format(max_entities=self.max_entities)
        try:
            result = await self.chat_json(
                [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": _USER_TMPL.format(text=text[:4000]),
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw: list[dict[str, str]] = result.get("entities", [])
            return [
                ChemicalEntity(
                    mention=e.get("mention", ""),
                    normalized_name=e.get("normalized_name", e.get("mention", "")),
                    entity_type=e.get("entity_type", "compound"),
                )
                for e in raw
                if e.get("mention")
            ][: self.max_entities]
        except Exception as exc:
            log.warning("Entity extraction failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # PubChem linking
    # ------------------------------------------------------------------

    async def _fetch_cid(self, session: aiohttp.ClientSession, name: str) -> Optional[int]:
        url = f"{_PUBCHEM_BASE}/compound/name/{aiohttp.helpers.quote(name)}/cids/JSON"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
                cids = data.get("IdentifierList", {}).get("CID", [])
                return cids[0] if cids else None
        except Exception as exc:
            log.debug("PubChem CID lookup failed for '%s': %s", name, exc)
            return None

    async def _fetch_properties(
        self, session: aiohttp.ClientSession, cid: int
    ) -> dict[str, str]:
        props = "IUPACName,MolecularFormula,InChIKey,CanonicalSMILES"
        url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/{props}/JSON"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {}
                data = await resp.json(content_type=None)
                table = data.get("PropertyTable", {}).get("Properties", [{}])
                return table[0] if table else {}
        except Exception as exc:
            log.debug("PubChem property fetch failed for CID %d: %s", cid, exc)
            return {}

    async def _link_entity(
        self, session: aiohttp.ClientSession, entity: ChemicalEntity
    ) -> ChemicalEntity:
        cid = await self._fetch_cid(session, entity.normalized_name)
        if cid is None:
            # Fallback: try the raw mention text
            cid = await self._fetch_cid(session, entity.mention)
        if cid is not None:
            props = await self._fetch_properties(session, cid)
            entity.cid = cid
            entity.iupac_name = props.get("IUPACName", "")
            entity.molecular_formula = props.get("MolecularFormula", "")
            entity.inchikey = props.get("InChIKey", "")
            entity.canonical_smiles = props.get("CanonicalSMILES", "")
            entity.pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
        return entity

    async def link(self, entities: list[ChemicalEntity]) -> list[ChemicalEntity]:
        """Resolve PubChem CIDs + properties for all *entities* concurrently."""
        if not entities:
            return []
        async with aiohttp.ClientSession(timeout=self.pubchem_timeout) as session:
            linked = await asyncio.gather(
                *[self._link_entity(session, e) for e in entities],
                return_exceptions=True,
            )
        result: list[ChemicalEntity] = []
        for item in linked:
            if isinstance(item, Exception):
                log.warning("Entity linking error: %s", item)
            else:
                result.append(item)
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, novelty_results: list[NoveltyResult]) -> dict[str, list[ChemicalEntity]]:
        """Extract and link entities for a list of selected papers.

        Returns a mapping of paper_id → list[ChemicalEntity].
        """
        output: dict[str, list[ChemicalEntity]] = {}
        for nr in novelty_results:
            log.info("Extracting entities from %s …", nr.paper_id)
            raw_entities = await self.extract_entities(nr.best_segment_text)
            linked = await self.link(raw_entities)
            linked_count = sum(1 for e in linked if e.cid is not None)
            log.info(
                "  %d entities extracted, %d linked to PubChem",
                len(linked),
                linked_count,
            )
            output[nr.paper_id] = linked
        return output
