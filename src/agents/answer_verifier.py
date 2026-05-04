"""
Answer Verifier

Verifies the ground-truth answers stored in the benchmark JSON by actually running
RDKit (for T2 and structural T3) and PubChem API calls (for T1).

Flags any question whose computed answer does not match the stored answer so it can
be corrected before the benchmark is used for LLM evaluation.

No LLM calls — pure deterministic computation.

Usage:
    python -m src.agents.answer_verifier
    python -m src.agents.answer_verifier --benchmark data/benchmark/livechembench_v0.1.0.json
    python -m src.agents.answer_verifier --question-id lcb_0001
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

from src.agents.models import (
    BenchmarkQuestion,
    LiveChemBench,
    QuestionType,
    VerificationReport,
    VerificationResult,
    VerificationStatus,
)
from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

# ---------------------------------------------------------------------------
# SMILES / CID extraction helpers
# ---------------------------------------------------------------------------

def _extract_smiles_balanced(text: str, start: int) -> str:
    """Extract SMILES starting at position `start`, stopping at the matching ')'.
    Handles nested parentheses within SMILES strings."""
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return text[start : i - 1]  # exclude the final closing paren


def _extract_smiles_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (name, SMILES) pairs from patterns like 'Compound (SMILES: ...)'."""
    results = []
    name_pat = re.compile(r"([\w][\w\s'\"(),/-]*?)\s*\(SMILES:\s*", re.IGNORECASE)
    for m in name_pat.finditer(text):
        name = m.group(1).strip()
        smiles = _extract_smiles_balanced(text, m.end())
        results.append((name, smiles))
    return results


def _extract_first_smiles(text: str) -> Optional[str]:
    m = re.search(r"\(SMILES:\s*", text, re.IGNORECASE)
    if not m:
        return None
    return _extract_smiles_balanced(text, m.end())


def _extract_cid(text: str) -> Optional[int]:
    m = re.search(r"PubChem CID[:\s]+(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _detect_property(text: str) -> Optional[str]:
    q = text.lower()
    if "rotatable bond" in q:       return "rotatable_bonds"
    if "aromatic atom" in q:        return "aromatic_atoms"
    if "aromatic ring" in q:        return "aromatic_rings"
    if "number of ring" in q or ("ring" in q and "count" in q): return "ring_count"
    if "h-bond donor" in q or "hydrogen bond donor" in q:       return "hbd"
    if "h-bond acceptor" in q or "hydrogen bond acceptor" in q: return "hba"
    if "molecular formula" in q:    return "molecular_formula"
    if "monoisotopic mass" in q or "exact mass" in q:           return "monoisotopic_mass"
    if "xlogp" in q or "logp" in q: return "xlogp"
    if "tpsa" in q:                 return "tpsa"
    if "molecular weight" in q:     return "molecular_weight"
    return None

# ---------------------------------------------------------------------------
# RDKit computation
# ---------------------------------------------------------------------------

def _rdkit_compute(smiles: str, prop: str) -> tuple[str, Optional[str]]:
    """Return (value_str, error). Imports RDKit lazily."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except ImportError:
        return "", "RDKit not installed"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", f"Invalid SMILES: {smiles[:60]}"

    try:
        if prop == "rotatable_bonds":
            val = rdMolDescriptors.CalcNumRotatableBonds(mol)
        elif prop == "aromatic_atoms":
            val = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        elif prop == "aromatic_rings":
            ri = mol.GetRingInfo()
            val = sum(1 for ring in ri.AtomRings()
                      if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        elif prop == "ring_count":
            val = mol.GetRingInfo().NumRings()
        elif prop == "hbd":
            val = rdMolDescriptors.CalcNumHBD(mol)
        elif prop == "hba":
            val = rdMolDescriptors.CalcNumHBA(mol)
        elif prop == "molecular_formula":
            val = rdMolDescriptors.CalcMolFormula(mol)
        elif prop == "monoisotopic_mass":
            val = round(Descriptors.ExactMolWt(mol), 4)
        elif prop == "molecular_weight":
            val = round(Descriptors.MolWt(mol), 4)
        else:
            return "", f"No RDKit handler for property: {prop}"
        return str(val), None
    except Exception as exc:
        return "", str(exc)

# ---------------------------------------------------------------------------
# PubChem API
# ---------------------------------------------------------------------------

_PUBCHEM_PROP_MAP = {
    "molecular_formula":  "MolecularFormula",
    "monoisotopic_mass":  "MonoisotopicMass",
    "xlogp":              "XLogP",
    "tpsa":               "TPSA",
    "molecular_weight":   "MolecularWeight",
}


def _pubchem_fetch(identifier: str, by: str, prop_key: str) -> tuple[str, Optional[str]]:
    """
    Fetch a single property from PubChem.
    by = 'cid' | 'name' | 'smiles'
    Returns (value_str, error).
    """
    pubchem_prop = _PUBCHEM_PROP_MAP.get(prop_key)
    if not pubchem_prop:
        return "", f"No PubChem mapping for property: {prop_key}"

    if by == "smiles":
        url = f"{_PUBCHEM_BASE}/smiles/{identifier}/property/{pubchem_prop}/JSON"
    elif by == "cid":
        url = f"{_PUBCHEM_BASE}/cid/{identifier}/property/{pubchem_prop}/JSON"
    else:
        encoded = identifier.replace(" ", "%20").replace("'", "%27")
        url = f"{_PUBCHEM_BASE}/name/{encoded}/property/{pubchem_prop}/JSON"

    try:
        time.sleep(0.2)  # respect PubChem rate limit
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        props = data["PropertyTable"]["Properties"][0]
        val = props.get(pubchem_prop)
        if val is None:
            return "", f"PubChem returned no value for {pubchem_prop}"
        # Round monoisotopic mass to 4 decimal places to match benchmark format
        if prop_key == "monoisotopic_mass":
            val = round(float(val), 4)
        return str(val), None
    except URLError as exc:
        return "", f"PubChem network error: {exc}"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return "", f"PubChem parse error: {exc}"

# ---------------------------------------------------------------------------
# Per-question-type verifiers
# ---------------------------------------------------------------------------

def _verify_t1(q: BenchmarkQuestion) -> tuple[str, Optional[str]]:
    prop = _detect_property(q.question_text)
    if not prop:
        return "", "Could not detect property from question text"

    smiles = _extract_first_smiles(q.question_text)
    cid = _extract_cid(q.question_text)

    # Prefer RDKit for properties it can compute reliably from SMILES
    rdkit_props = {"rotatable_bonds", "aromatic_atoms", "aromatic_rings",
                   "ring_count", "hbd", "hba", "molecular_formula", "monoisotopic_mass"}

    if smiles and prop in rdkit_props:
        return _rdkit_compute(smiles, prop)

    if smiles and prop in _PUBCHEM_PROP_MAP:
        from urllib.parse import quote
        return _pubchem_fetch(quote(smiles, safe=""), "smiles", prop)

    if cid and prop in _PUBCHEM_PROP_MAP:
        return _pubchem_fetch(str(cid), "cid", prop)

    # Fall back to compound name lookup
    for entity in q.chemical_entities:
        if prop in _PUBCHEM_PROP_MAP:
            val, err = _pubchem_fetch(entity, "name", prop)
            if not err:
                return val, None

    return "", f"Cannot verify T1: no SMILES, CID, or usable entity found"


def _verify_t2(q: BenchmarkQuestion) -> tuple[str, Optional[str]]:
    prop = _detect_property(q.question_text)
    if not prop:
        return "", "Could not detect property from question text"

    smiles = _extract_first_smiles(q.question_text)
    if not smiles:
        # Try chemical_entities for a SMILES string
        for entity in q.chemical_entities:
            if any(c in entity for c in ("=", "(", ")", "[", "#")):
                smiles = entity
                break

    if not smiles:
        return "", "No SMILES found in question text or chemical_entities"

    return _rdkit_compute(smiles, prop)


def _verify_t3(q: BenchmarkQuestion) -> tuple[str, Optional[str]]:
    prop = _detect_property(q.question_text)
    if not prop:
        return "", "Could not detect property"

    pairs = _extract_smiles_pairs(q.question_text)
    if len(pairs) < 2:
        return "", f"Expected 2 compound-SMILES pairs, found {len(pairs)}"

    computed: list[tuple[str, float]] = []
    for name, smiles in pairs[:2]:
        val_str, err = _rdkit_compute(smiles.strip(), prop)
        if err:
            return "", f"RDKit failed for '{name}': {err}"
        try:
            computed.append((name.strip(), float(val_str)))
        except ValueError:
            return "", f"Non-numeric RDKit result for '{name}': {val_str}"

    q_lower = q.question_text.lower()
    if any(w in q_lower for w in ("higher", "more", "greater", "larger")):
        winner_name, _ = max(computed, key=lambda x: x[1])
    elif any(w in q_lower for w in ("lower", "fewer", "less", "smaller")):
        winner_name, _ = min(computed, key=lambda x: x[1])
    else:
        return "", "Cannot determine comparison direction (higher/lower/more/fewer)"

    return winner_name, None

# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _answers_match(computed: str, expected: str, q: BenchmarkQuestion) -> bool:
    from src.agents.models import AnswerType
    computed = computed.strip()
    expected = expected.strip()

    if q.answer_type == AnswerType.float_:
        try:
            tol = q.tolerance if q.tolerance is not None else 0.01
            return abs(float(computed) - float(expected)) <= tol
        except ValueError:
            return False
    elif q.answer_type == AnswerType.int_:
        try:
            return int(computed) == int(expected)
        except ValueError:
            return computed.lower() == expected.lower()
    else:
        # string / choice — case-insensitive, also accept substring match for names
        if computed.lower() == expected.lower():
            return True
        # allow "Nec-1s" to match "Nec-1s (SMILES: ...)" style expected
        return expected.lower() in computed.lower() or computed.lower() in expected.lower()

# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------

def verify(
    benchmark: LiveChemBench,
    question_id: Optional[str] = None,
) -> VerificationReport:
    results: list[VerificationResult] = []
    questions = benchmark.questions
    if question_id:
        questions = [q for q in questions if q.id == question_id]

    for q in questions:
        logger.info("Verifying %s [%s] ...", q.id, q.question_type.value)
        computed, error = "", None

        if q.question_type == QuestionType.T1:
            computed, error = _verify_t1(q)
        elif q.question_type == QuestionType.T2:
            computed, error = _verify_t2(q)
        elif q.question_type == QuestionType.T3:
            computed, error = _verify_t3(q)

        if error:
            status = VerificationStatus.error
            logger.warning("  %s ERROR: %s", q.id, error)
        elif _answers_match(computed, q.answer, q):
            status = VerificationStatus.correct
            logger.info("  %s CORRECT: computed=%r expected=%r", q.id, computed, q.answer)
        else:
            status = VerificationStatus.wrong
            logger.warning(
                "  %s WRONG: computed=%r expected=%r", q.id, computed, q.answer
            )

        results.append(VerificationResult(
            question_id=q.id,
            question_type=q.question_type,
            expected_answer=q.answer,
            computed_answer=computed or None,
            status=status,
            error=error,
        ))

    summary: dict[str, int] = defaultdict(int)
    for r in results:
        summary[r.status.value] += 1

    logger.info(
        "Verification complete: %s", dict(summary)
    )
    return VerificationReport(
        benchmark_version=benchmark.version,
        verified_at=datetime.now(timezone.utc).isoformat(),
        results=results,
        summary=dict(summary),
    )


def run(benchmark_path: Path, output_dir: Path, question_id: Optional[str] = None) -> VerificationReport:
    benchmark = LiveChemBench.model_validate_json(benchmark_path.read_text())
    report = verify(benchmark, question_id=question_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"livechembench_v{benchmark.version}_verified.json"
    out_path.write_text(report.model_dump_json(indent=2))
    logger.info("Verification report written to: %s", out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify ground-truth answers in the benchmark using RDKit and PubChem."
    )
    parser.add_argument(
        "--benchmark",
        default=str(_REPO_ROOT / "data" / "benchmark" / "livechembench_v0.1.0.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "data" / "verification"),
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="Verify a single question by ID (e.g. lcb_0001).",
    )
    args = parser.parse_args()

    run(
        benchmark_path=Path(args.benchmark),
        output_dir=Path(args.output_dir),
        question_id=args.question_id,
    )


if __name__ == "__main__":
    main()
