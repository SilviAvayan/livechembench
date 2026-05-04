"""
A3 — PubChem Linker

Resolves chemical entity names from segmented papers to canonical PubChem CIDs.
For each resolved entity, fetches key properties (exact mass, molecular formula,
canonical SMILES, XLogP3, TPSA, H-bond donors/acceptors, rotatable bonds).

Outputs: data/pubchem_links/<paper_id>.json

Usage:
    python -m src.agents.pubchem_linker                   # all worthy papers
    python -m src.agents.pubchem_linker --paper-id <id>   # single paper
    python -m src.agents.pubchem_linker --limit 5         # first 5 papers
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Properties to fetch for each resolved compound
_PROPERTIES = ",".join([
    "MolecularFormula",
    "MolecularWeight",
    "ExactMass",
    "CanonicalSMILES",
    "IsomericSMILES",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "HeavyAtomCount",
    "Complexity",
    "IUPACName",
])

_REQUEST_DELAY = 0.22  # PubChem rate limit: ~5 req/s


def _fetch_cid_by_name(name: str) -> Optional[int]:
    """Return the best-match PubChem CID for a compound name, or None."""
    url = f"{_PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    try:
        resp = requests.get(url, timeout=10)
        time.sleep(_REQUEST_DELAY)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        return cids[0] if cids else None
    except Exception as exc:
        logger.debug("PubChem name lookup failed for %r: %s", name, exc)
        return None


def _fetch_properties(cid: int) -> dict[str, Any]:
    """Fetch standard properties for a CID."""
    url = f"{_PUBCHEM_BASE}/compound/cid/{cid}/property/{_PROPERTIES}/JSON"
    try:
        resp = requests.get(url, timeout=10)
        time.sleep(_REQUEST_DELAY)
        resp.raise_for_status()
        props_list = resp.json().get("PropertyTable", {}).get("Properties", [])
        return props_list[0] if props_list else {}
    except Exception as exc:
        logger.debug("PubChem property fetch failed for CID %d: %s", cid, exc)
        return {}


def _extract_entities(paper: dict) -> list[str]:
    """
    Extract candidate chemical entity names from a segmented paper.
    Pulls from: abstract, key_points, tables (first column), and conclusion.
    """
    entities: set[str] = set()

    # Named compound patterns: capitalised words that look like chemical names
    # We look in text-heavy fields for parenthetical names, compound codes, etc.
    text_fields = [
        paper.get("abstract") or "",
        paper.get("conclusion") or "",
        " ".join(paper.get("key_points") or []),
    ]
    # Include table cell content — entries may be dicts with "rows" or plain strings
    for table in paper.get("tables") or []:
        if isinstance(table, str):
            text_fields.append(table)
        elif isinstance(table, dict):
            for row in table.get("rows") or []:
                if row:
                    entities.add(str(row[0]).strip())

    full_text = " ".join(text_fields)

    # Patterns for chemical names: compounds like GYY4137, andrographolide,
    # Nec-1s, etc. We pull parenthetical names and CamelCase/hyphen names.
    patterns = [
        # Parenthetical compound names: "(compound name)" or "(Compound Name)"
        r"\(([A-Z][a-zA-Z0-9\-\']+(?:\s[a-zA-Z0-9\-]+){0,4})\)",
        # Compound codes: alphanumeric codes like GYY4137, NSC-12345, etc.
        r"\b([A-Z]{2,}[\'\-]?\d{2,}[a-zA-Z]?)\b",
        # Compound codes like Nec-1s, GSK'872
        r"\b([A-Z][a-zA-Z]{1,4}[\'\-]\d{1,4}[a-zA-Z]?)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, full_text):
            name = m.group(1).strip()
            if len(name) >= 3:
                entities.add(name)

    # Also add any items already tagged as chemical entities in key_points if
    # they look like compound names (heuristic: CamelCase or contains digits)
    for kp in paper.get("key_points") or []:
        words = kp.split()
        for word in words:
            word_clean = word.strip(".,;:()")
            if re.match(r"^[A-Z][a-zA-Z]*[0-9][a-zA-Z0-9\-\']*$", word_clean):
                entities.add(word_clean)

    return sorted(entities)


def link_paper(paper: dict) -> dict:
    """
    Resolve chemical entities in a paper to PubChem CIDs with properties.
    Returns a dict with paper_id, resolved entities, and unresolved names.
    """
    paper_id = paper.get("paper_id", "unknown")
    candidates = _extract_entities(paper)

    resolved: list[dict] = []
    unresolved: list[str] = []

    logger.info("  Linking %d candidate entities for %s", len(candidates), paper_id)

    for name in candidates:
        # Skip obvious non-chemical strings
        if len(name) < 3 or name.lower() in {
            "the", "and", "for", "with", "from", "that", "this",
            "fig", "table", "figure", "method", "result", "data",
        }:
            continue

        cid = _fetch_cid_by_name(name)
        if cid is None:
            unresolved.append(name)
            logger.debug("  Unresolved: %r", name)
            continue

        props = _fetch_properties(cid)
        if not props:
            unresolved.append(name)
            continue

        resolved.append({
            "queried_name": name,
            "cid": cid,
            "iupac_name": props.get("IUPACName"),
            "molecular_formula": props.get("MolecularFormula"),
            "molecular_weight": props.get("MolecularWeight"),
            "exact_mass": props.get("ExactMass"),
            "canonical_smiles": props.get("CanonicalSMILES"),
            "isomeric_smiles": props.get("IsomericSMILES"),
            "xlogp": props.get("XLogP"),
            "tpsa": props.get("TPSA"),
            "hbd": props.get("HBondDonorCount"),
            "hba": props.get("HBondAcceptorCount"),
            "rotatable_bonds": props.get("RotatableBondCount"),
            "heavy_atom_count": props.get("HeavyAtomCount"),
        })
        logger.debug("  Resolved %r → CID %d (%s)", name, cid, props.get("MolecularFormula", "?"))

    return {
        "paper_id": paper_id,
        "resolved": resolved,
        "unresolved": unresolved,
        "linked_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    segmented_dir: Path,
    evaluations_dir: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only process papers marked worth_pursuing
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

    logger.info("PubChem Linker: processing %d paper(s)", len(candidates))

    for json_path in candidates:
        if not json_path.exists():
            logger.warning("File not found, skipping: %s", json_path)
            continue

        out_path = output_dir / json_path.name
        if out_path.exists():
            logger.info("Already linked, skipping: %s", json_path.stem)
            continue

        try:
            paper = json.loads(json_path.read_text())
        except Exception as exc:
            logger.error("Failed to load %s: %s", json_path.stem, exc)
            continue

        logger.info("Linking: %s", json_path.stem)
        result = link_paper(paper)
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(
            "  → %d resolved, %d unresolved for %s",
            len(result["resolved"]),
            len(result["unresolved"]),
            json_path.stem,
        )

    logger.info("Done. Results in: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A3: Resolve chemical entities to PubChem CIDs."
    )
    parser.add_argument("--paper-id", default=None, help="Link a single paper by ID.")
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
        default=str(_REPO_ROOT / "data" / "pubchem_links"),
    )
    args = parser.parse_args()

    run(
        segmented_dir=Path(args.segmented_dir),
        evaluations_dir=Path(args.evaluations_dir),
        output_dir=Path(args.output_dir),
        paper_id=args.paper_id,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
