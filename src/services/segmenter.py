"""
Segmenter: Parses PDFs and extracts abstract, key points, conclusion, tables, and figures.

Engines:
  - paddle_vl: PaddleOCR-VL-1.5 document parser (default). Produces full-document
    markdown via restructure_pages, table text from layout blocks, and saves figure
    crops from markdown image payloads.
  - docling: IBM Docling PDF→markdown + the same heading-based section heuristics
    (no structured table list or figure exports).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.config.models import SegmentationConfig
from src.utils.logger import logger

# Project root (…/livechembench)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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
    tables: List[str]
    figure_paths: List[str]
    extraction_status: str  # "success" | "partial" | "failed"
    section_count: int
    raw_char_count: int
    compressed_char_count: int
    engine: str = "paddle_vl"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Heading / section helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _heading_normalised(raw: str) -> str:
    text = raw.lower().strip()
    text = re.sub(r"^[\d\.\s]+", "", text)
    text = re.sub(r"^[ivxlcdm]+\.\s+", "", text)
    return text.strip(" .")


def _is_heading_match(raw_heading: str, keywords: List[str]) -> bool:
    norm = _heading_normalised(raw_heading)
    return any(kw in norm for kw in keywords)


def _split_into_sections(markdown: str) -> List[tuple[str, str]]:
    sections: List[tuple[str, str]] = []
    last_heading = ""
    last_pos = 0

    for m in _HEADING_RE.finditer(markdown):
        body = markdown[last_pos : m.start()].strip()
        if body or last_heading:
            sections.append((last_heading, body))
        last_heading = m.group(1)
        last_pos = m.end()

    body = markdown[last_pos:].strip()
    if body or last_heading:
        sections.append((last_heading, body))

    return sections


def _first_n_sentences(text: str, n: int = 2) -> str:
    sentences = _SENT_RE.split(text.strip())[:n]
    joined = " ".join(s.strip() for s in sentences if s.strip())
    return joined


_SKIP_SECTION_KEYWORDS = {
    "reference",
    "references",
    "bibliography",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "funding",
    "conflicts of interest",
    "conflict of interest",
    "author contribution",
    "author contributions",
    "supplementary",
    "supporting information",
    "appendix",
    "data availability",
    "ethics",
    "declaration",
}


def _is_skip_section(heading: str) -> bool:
    norm = _heading_normalised(heading)
    return any(kw in norm for kw in _SKIP_SECTION_KEYWORDS)


def _sections_from_markdown(
    markdown: str, cfg: SegmentationConfig, title: str
) -> tuple[str, str, List[str], str, str, int]:
    """
    Shared heuristic: markdown → abstract, key_points, conclusion, status, section_count.
    """
    abstract = ""
    key_points: List[str] = []
    conclusion = ""
    status = "success"

    sections = _split_into_sections(markdown)
    section_count = len(sections)

    if not sections:
        return title, abstract, key_points, conclusion, "partial", section_count

    abstract_kws = [h.lower() for h in cfg.abstract_headings]
    conclusion_kws = [h.lower() for h in cfg.conclusion_headings]

    for heading, body in sections:
        if _is_heading_match(heading, abstract_kws):
            abstract = body.strip()
            break

    if not abstract:
        for heading, body in sections:
            if not heading and len(body) >= cfg.min_section_chars:
                abstract = body.strip()
                break
        if not abstract and sections:
            for _, body in sections:
                if len(body) >= cfg.min_section_chars:
                    abstract = body.strip()
                    break

    for heading, body in reversed(sections):
        if _is_heading_match(heading, conclusion_kws):
            conclusion = body.strip()
            break

    abstract_headings_set = set(abstract_kws)
    conclusion_headings_set = set(conclusion_kws)

    middle_bodies: List[str] = []
    for heading, body in sections:
        norm = _heading_normalised(heading)
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

    has_abstract = bool(abstract)
    has_conclusion = bool(conclusion)
    has_key_points = bool(key_points)

    if has_abstract and has_conclusion and has_key_points:
        status = "success"
    elif has_abstract or has_conclusion or has_key_points:
        status = "partial"
    else:
        status = "partial"

    return title, abstract, key_points, conclusion, status, section_count


# ---------------------------------------------------------------------------
# PaddleOCR-VL helpers
# ---------------------------------------------------------------------------


def _paddle_inner_json(res: Any) -> Dict[str, Any]:
    j = getattr(res, "json", None)
    if j is None:
        return {}
    if isinstance(j, dict) and "res" in j:
        inner = j.get("res")
        return inner if isinstance(inner, dict) else {}
    return j if isinstance(j, dict) else {}


def _markdown_text_from_result(res: Any) -> str:
    """Prefer the `markdown` dict; fall back to joining parsing blocks."""
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        texts = md.get("markdown_texts")
        if isinstance(texts, str) and texts.strip():
            return texts
        if isinstance(texts, (list, tuple)):
            joined = "\n\n".join(str(t).strip() for t in texts if t)
            if joined.strip():
                return joined
    inner = _paddle_inner_json(res)
    parts: List[str] = []
    for block in inner.get("parsing_res_list") or []:
        if not isinstance(block, dict):
            continue
        label = (block.get("block_label") or "").lower()
        if label in ("text", "paragraph_title", "doc_title", "header", "footer"):
            c = block.get("block_content")
            if c:
                parts.append(str(c))
    return "\n\n".join(parts)


def _merge_markdown_from_page_results(results: Sequence[Any]) -> str:
    chunks: List[str] = []
    for res in results:
        t = _markdown_text_from_result(res)
        if t.strip():
            chunks.append(t.strip())
    return "\n\n".join(chunks)


def _iter_parsing_blocks(results: Sequence[Any]) -> Iterable[Dict[str, Any]]:
    for res in results:
        inner = _paddle_inner_json(res)
        for block in inner.get("parsing_res_list") or []:
            if isinstance(block, dict):
                yield block


def _collect_tables(
    results: Sequence[Any], table_labels: Sequence[str]
) -> List[str]:
    labels = {x.lower() for x in table_labels}
    out: List[str] = []
    for block in _iter_parsing_blocks(results):
        lab = (block.get("block_label") or "").lower()
        if lab not in labels:
            continue
        content = block.get("block_content")
        if content and str(content).strip():
            out.append(str(content).strip())
    return out


def _pil_image_from_markdown_item(item: Any) -> Any:
    """Resolve a markdown image slot to a PIL Image (Paddle may use Image or nested dict)."""
    if item is None:
        return None
    if hasattr(item, "save") and callable(getattr(item, "save", None)):
        return item
    if isinstance(item, dict):
        for key in ("image", "img", "pil", "figure", "data"):
            inner = item.get(key)
            if inner is not None and hasattr(inner, "save"):
                return inner
    return None


def _save_markdown_figures(results: Sequence[Any], assets_dir: Path) -> List[str]:
    """Save PIL images from `markdown.markdown_images` and return repo-relative paths."""
    saved: List[str] = []
    idx = 0
    assets_dir.mkdir(parents=True, exist_ok=True)

    for res in results:
        md = getattr(res, "markdown", None)
        if not isinstance(md, dict):
            continue
        images = md.get("markdown_images")
        if images is None:
            continue
        if not isinstance(images, (list, tuple)):
            images = [images]
        for raw in images:
            img = _pil_image_from_markdown_item(raw)
            if img is None:
                continue
            path = assets_dir / f"figure_{idx:03d}.png"
            idx += 1
            try:
                img.save(path)
                try:
                    rel = path.relative_to(_REPO_ROOT)
                    saved.append(str(rel))
                except ValueError:
                    saved.append(str(path))
            except Exception as exc:
                logger.warning(f"Could not save figure to {path}: {exc}")

    return saved


def segment_paper_paddle_vl(
    pdf_path: Path,
    cfg: SegmentationConfig,
    assets_root: Path,
    pipeline: Any = None,
) -> SegmentedPaper:
    paper_id = pdf_path.stem
    source_file = pdf_path.name
    pcfg = cfg.paddle_vl

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        logger.error(
            "PaddleOCR is not installed. Install optional deps: "
            "pip install -r requirements-paddleocr-vl.txt (%s)",
            exc,
        )
        return SegmentedPaper(
            paper_id=paper_id,
            source_file=source_file,
            title="",
            abstract="",
            key_points=[],
            conclusion="",
            tables=[],
            figure_paths=[],
            extraction_status="failed",
            section_count=0,
            raw_char_count=0,
            compressed_char_count=0,
            engine="paddle_vl",
        )

    if pipeline is None:
        kwargs: Dict[str, Any] = {"pipeline_version": pcfg.pipeline_version}
        if pcfg.device:
            kwargs["device"] = pcfg.device
        pipeline = PaddleOCRVL(**kwargs)

    pages_res = list(pipeline.predict(input=str(pdf_path)))

    if not pages_res:
        logger.error(f"PaddleOCR-VL returned no pages for {source_file}")
        return SegmentedPaper(
            paper_id=paper_id,
            source_file=source_file,
            title="",
            abstract="",
            key_points=[],
            conclusion="",
            tables=[],
            figure_paths=[],
            extraction_status="failed",
            section_count=0,
            raw_char_count=0,
            compressed_char_count=0,
            engine="paddle_vl",
        )

    restructured = pipeline.restructure_pages(
        pages_res,
        merge_tables=pcfg.merge_tables,
        relevel_titles=pcfg.relevel_titles,
        concatenate_pages=pcfg.concatenate_pages,
    )
    results_for_md = list(restructured) if restructured else pages_res

    markdown = _merge_markdown_from_page_results(results_for_md)
    if not markdown.strip():
        markdown = _merge_markdown_from_page_results(pages_res)

    raw_char_count = len(markdown)

    title = ""
    first = results_for_md[0] if results_for_md else pages_res[0]
    inner = _paddle_inner_json(first)
    for block in inner.get("parsing_res_list") or []:
        if not isinstance(block, dict):
            continue
        if (block.get("block_label") or "").lower() == "doc_title":
            title = str(block.get("block_content") or "").strip()
            break

    tables = _collect_tables(results_for_md or pages_res, pcfg.table_labels)
    assets_dir = assets_root / paper_id
    figure_paths = _save_markdown_figures(results_for_md or pages_res, assets_dir)

    t, abstract, key_points, conclusion, status, section_count = _sections_from_markdown(
        markdown, cfg, title
    )

    compressed_char_count = (
        len(abstract)
        + sum(len(p) for p in key_points)
        + len(conclusion)
        + sum(len(t) for t in tables)
    )

    return SegmentedPaper(
        paper_id=paper_id,
        source_file=source_file,
        title=t,
        abstract=abstract,
        key_points=key_points,
        conclusion=conclusion,
        tables=tables,
        figure_paths=figure_paths,
        extraction_status=status,
        section_count=section_count,
        raw_char_count=raw_char_count,
        compressed_char_count=compressed_char_count,
        engine="paddle_vl",
    )


# ---------------------------------------------------------------------------
# Docling engine
# ---------------------------------------------------------------------------


def segment_paper_docling(pdf_path: Path, cfg: SegmentationConfig) -> SegmentedPaper:
    paper_id = pdf_path.stem
    source_file = pdf_path.name
    title = ""
    markdown = ""
    raw_char_count = 0

    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        raw_char_count = len(markdown)
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
            tables=[],
            figure_paths=[],
            extraction_status="failed",
            section_count=0,
            raw_char_count=0,
            compressed_char_count=0,
            engine="docling",
        )

    t, abstract, key_points, conclusion, status, section_count = _sections_from_markdown(
        markdown, cfg, title
    )

    compressed_char_count = len(abstract) + sum(len(p) for p in key_points) + len(conclusion)

    return SegmentedPaper(
        paper_id=paper_id,
        source_file=source_file,
        title=t,
        abstract=abstract,
        key_points=key_points,
        conclusion=conclusion,
        tables=[],
        figure_paths=[],
        extraction_status=status,
        section_count=section_count,
        raw_char_count=raw_char_count,
        compressed_char_count=compressed_char_count,
        engine="docling",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_paper(
    pdf_path: Path,
    cfg: SegmentationConfig,
    assets_root: Optional[Path] = None,
    paddle_pipeline: Any = None,
    layout_root: Optional[Path] = None,
    layout_model: Any = None,
) -> SegmentedPaper:
    pdf_path = Path(pdf_path)
    if cfg.engine == "docling":
        return segment_paper_docling(pdf_path, cfg)
    root = assets_root or (_REPO_ROOT / "data" / "segmented_papers" / "assets")
    return segment_paper_paddle_vl(
        pdf_path, cfg, root.resolve(), pipeline=paddle_pipeline
    )
