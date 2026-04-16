"""
Segmenter: Uses IBM Docling to parse PDFs and extract structured sections.

Outputs per paper:
  - abstract    : text of identified abstract section
  - key_points  : list of concise points from middle sections (intro, methods, results)
  - conclusion  : text of identified conclusion section
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from src.config.models import SegmentationConfig
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SegmentedPaper:
    paper_id: str
    source_file: str
    title: str
    abstract: str
    key_points: List[str]
    conclusion: str
    extraction_status: str          # "success" | "partial" | "failed"
    section_count: int
    raw_char_count: int
    compressed_char_count: int

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Heading / section helpers
# ---------------------------------------------------------------------------

# Matches Markdown headings: ## Abstract, # 1. Introduction, etc.
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)

# Sentence boundary — crude but fast: split on ". " followed by capital letter
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _heading_normalised(raw: str) -> str:
    """Lowercase, strip numbers and trailing punctuation for robust matching."""
    text = raw.lower().strip()
    # Remove leading numbering like "1.", "2.1", "III."
    text = re.sub(r"^[\d\.\s]+", "", text)
    text = re.sub(r"^[ivxlcdm]+\.\s+", "", text)  # Roman numerals
    return text.strip(" .")


def _is_heading_match(raw_heading: str, keywords: List[str]) -> bool:
    norm = _heading_normalised(raw_heading)
    return any(kw in norm for kw in keywords)


def _split_into_sections(markdown: str) -> List[tuple[str, str]]:
    """
    Returns a list of (heading, body) pairs.
    Section with no heading has heading = "" (preamble text before first heading).
    """
    sections: List[tuple[str, str]] = []
    last_heading = ""
    last_pos = 0

    for m in _HEADING_RE.finditer(markdown):
        body = markdown[last_pos:m.start()].strip()
        if body or last_heading:
            sections.append((last_heading, body))
        last_heading = m.group(1)
        last_pos = m.end()

    # Final section
    body = markdown[last_pos:].strip()
    if body or last_heading:
        sections.append((last_heading, body))

    return sections


def _first_n_sentences(text: str, n: int = 2) -> str:
    """Return the first `n` sentences of the given text block."""  # noqa: D401
    sentences = _SENT_RE.split(text.strip())[:n]
    joined = " ".join(s.strip() for s in sentences if s.strip())
    return joined


# ---------------------------------------------------------------------------
# Middle-section exclusion keywords (references, acknowledgements, etc.)
# ---------------------------------------------------------------------------

_SKIP_SECTION_KEYWORDS = {
    "reference", "references", "bibliography", "acknowledgement",
    "acknowledgements", "acknowledgment", "funding", "conflicts of interest",
    "conflict of interest", "author contribution", "author contributions",
    "supplementary", "supporting information", "appendix", "data availability",
    "ethics", "declaration",
}


def _is_skip_section(heading: str) -> bool:
    norm = _heading_normalised(heading)
    return any(kw in norm for kw in _SKIP_SECTION_KEYWORDS)


# ---------------------------------------------------------------------------
# Core segmentation function
# ---------------------------------------------------------------------------

def segment_paper(pdf_path: Path, cfg: SegmentationConfig) -> SegmentedPaper:
    """
    Parse `pdf_path` with Docling and extract abstract, key_points, conclusion.

    Falls back gracefully if Docling cannot parse or sections are missing.
    """
    paper_id = pdf_path.stem
    source_file = pdf_path.name
    title = ""
    abstract = ""
    key_points: List[str] = []
    conclusion = ""
    status = "success"
    raw_char_count = 0

    # ---- Parse with Docling ------------------------------------------------
    try:
        # Import lazily so the module can be imported even if docling is absent
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        raw_char_count = len(markdown)

        # Attempt to get title from Docling's built-in field (may be empty)
        doc_title = getattr(result.document, "name", "") or ""
        title = doc_title.strip()

    except Exception as exc:
        logger.error(f"Docling conversion failed for {source_file}: {exc}")
        return SegmentedPaper(
            paper_id=paper_id,
            source_file=source_file,
            title=title,
            abstract="",
            key_points=[],
            conclusion="",
            extraction_status="failed",
            section_count=0,
            raw_char_count=0,
            compressed_char_count=0,
        )

    # ---- Split into sections -----------------------------------------------
    sections = _split_into_sections(markdown)
    section_count = len(sections)

    if not sections:
        return SegmentedPaper(
            paper_id=paper_id,
            source_file=source_file,
            title=title,
            abstract="",
            key_points=[],
            conclusion="",
            extraction_status="partial",
            section_count=0,
            raw_char_count=raw_char_count,
            compressed_char_count=0,
        )

    # ---- Extract abstract --------------------------------------------------
    abstract_kws = [h.lower() for h in cfg.abstract_headings]
    conclusion_kws = [h.lower() for h in cfg.conclusion_headings]

    for heading, body in sections:
        if _is_heading_match(heading, abstract_kws):
            abstract = body.strip()
            break

    # Fallback: if no abstract heading found, use preamble (first section with
    # no recognised heading) that is long enough to be substantial
    if not abstract:
        for heading, body in sections:
            if not heading and len(body) >= cfg.min_section_chars:
                abstract = body.strip()
                break
        if not abstract and sections:
            # Last resort: use first non-empty section body regardless
            for _, body in sections:
                if len(body) >= cfg.min_section_chars:
                    abstract = body.strip()
                    break

    # ---- Extract conclusion ------------------------------------------------
    for heading, body in reversed(sections):
        if _is_heading_match(heading, conclusion_kws):
            conclusion = body.strip()
            break

    # ---- Extract key points from middle sections --------------------------
    abstract_headings_set = set(abstract_kws)
    conclusion_headings_set = set(conclusion_kws)

    middle_bodies: List[str] = []
    for heading, body in sections:
        norm = _heading_normalised(heading)
        # Skip abstract, conclusion itself, and noise sections
        is_abstract = any(kw in norm for kw in abstract_headings_set)
        is_conclusion = any(kw in norm for kw in conclusion_headings_set)
        if is_abstract or is_conclusion or _is_skip_section(heading):
            continue
        if len(body) >= cfg.min_section_chars:
            middle_bodies.append(body)

    for body in middle_bodies:
        if len(key_points) >= cfg.max_key_points:
            break
        point = _first_n_sentences(body, n=2)
        if point and len(point) >= 30:
            key_points.append(point)

    # ---- Determine extraction status ---------------------------------------
    has_abstract = bool(abstract)
    has_conclusion = bool(conclusion)
    has_key_points = bool(key_points)

    if has_abstract and has_conclusion and has_key_points:
        status = "success"
    elif has_abstract or has_conclusion or has_key_points:
        status = "partial"
    else:
        status = "partial"  # Still produced markdown, just no recognisable sections

    compressed_char_count = len(abstract) + sum(len(p) for p in key_points) + len(conclusion)

    return SegmentedPaper(
        paper_id=paper_id,
        source_file=source_file,
        title=title,
        abstract=abstract,
        key_points=key_points,
        conclusion=conclusion,
        extraction_status=status,
        section_count=section_count,
        raw_char_count=raw_char_count,
        compressed_char_count=compressed_char_count,
    )
