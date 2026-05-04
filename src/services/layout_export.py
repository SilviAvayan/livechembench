"""
Layout artifacts for PaddleOCR-VL: region JSON (bbox + label + text), colored overlays,
and a single extractive document summary from ordered regions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from src.utils.logger import logger

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Consistent colors for common PP-DocLayout / VL labels (outline + fill alpha applied later).
_LABEL_RGB: Dict[str, Tuple[int, int, int]] = {
    "doc_title": (220, 53, 69),
    "paragraph_title": (13, 110, 253),
    "text": (25, 135, 84),
    "abstract": (102, 16, 242),
    "abstract_title": (132, 94, 247),
    "content": (32, 201, 151),
    "content_title": (13, 202, 240),
    "table": (253, 126, 20),
    "table_title": (255, 193, 7),
    "figure_title": (214, 51, 132),
    "image": (111, 66, 193),
    "chart": (253, 126, 20),
    "reference": (108, 117, 125),
    "reference_title": (73, 80, 87),
    "reference_content": (134, 142, 150),
    "formula": (32, 201, 151),
    "display_formula": (25, 135, 84),
    "inline_formula": (32, 201, 151),
    "header": (173, 181, 189),
    "footer": (173, 181, 189),
    "footnote": (134, 142, 150),
    "ocr": (25, 135, 84),
    "number": (66, 66, 66),
    "seal": (214, 51, 132),
    "algorithm": (13, 110, 253),
}


def rgb_for_label(label: str) -> Tuple[int, int, int]:
    key = (label or "unknown").lower()
    if key in _LABEL_RGB:
        return _LABEL_RGB[key]
    h = abs(hash(key)) % (256**3)
    return (h % 180 + 30, (h // 256) % 180 + 30, (h // 65536) % 180 + 30)


def _paddle_inner_json(res: Any) -> Dict[str, Any]:
    j = getattr(res, "json", None)
    if callable(j):
        j = j()
    if j is None:
        return {}
    if isinstance(j, dict) and "res" in j:
        inner = j.get("res")
        return inner if isinstance(inner, dict) else {}
    return j if isinstance(j, dict) else {}


def _page_index_from_result(res: Any) -> int:
    for src in (res, _paddle_inner_json(res)):
        if not isinstance(src, dict) and hasattr(src, "__getitem__"):
            try:
                v = src["page_index"]  # type: ignore[index]
                if isinstance(v, list) and v:
                    v = v[0]
                return int(v)
            except Exception:
                pass
        if isinstance(src, dict):
            v = src.get("page_index")
            if v is not None:
                if isinstance(v, list) and v:
                    v = v[0]
                try:
                    return int(v)
                except Exception:
                    pass
    return 0


def _block_to_region_dict(block: Any) -> Optional[Dict[str, Any]]:
    if isinstance(block, dict):
        lab = block.get("block_label") or block.get("label")
        content = block.get("block_content")
        if content is None:
            content = block.get("content")
        bbox = block.get("block_bbox") or block.get("bbox") or block.get("box")
    else:
        lab = getattr(block, "label", None)
        content = getattr(block, "content", None)
        bbox = getattr(block, "bbox", None)
    if lab is None or bbox is None:
        return None
    bb = list(bbox)
    if len(bb) < 4:
        return None
    bb = [int(x) for x in bb[:4]]
    text = "" if content is None else str(content)
    return {
        "block_label": str(lab),
        "block_content": text,
        "block_bbox": bb,
    }


def _parsing_blocks_from_page_res(res: Any) -> List[Any]:
    blocks: List[Any] = []
    inner = _paddle_inner_json(res)
    pr = inner.get("parsing_res_list")
    if pr:
        blocks.extend(list(pr))
    if not blocks and hasattr(res, "__getitem__"):
        try:
            pl = res["parsing_res_list"]
            if pl:
                blocks.extend(list(pl))
        except Exception:
            pass
    return blocks


def collect_layout_regions(pages_res: Sequence[Any]) -> List[Dict[str, Any]]:
    """One dict per block: page_index, block_label, block_bbox, block_content."""
    out: List[Dict[str, Any]] = []
    for res in pages_res:
        page_idx = _page_index_from_result(res)
        for block in _parsing_blocks_from_page_res(res):
            d = _block_to_region_dict(block)
            if not d:
                continue
            row = dict(d)
            row["page_index"] = page_idx
            out.append(row)
    return out


def sort_regions_reading_order(regions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort blocks by page index, then top, then left (reading order)."""

    def key(r: Dict[str, Any]) -> Tuple[int, int, int]:
        bbox = r.get("block_bbox") or [0, 0, 0, 0]
        pi = int(r.get("page_index", 0))
        y = int(bbox[1]) if len(bbox) > 1 else 0
        x = int(bbox[0]) if bbox else 0
        return pi, y, x

    return sorted(regions, key=key)


def preprocessed_rgb_from_result(res: Any) -> Optional[Image.Image]:
    """RGB PIL image aligned with layout bbox coordinates (doc preprocessor output)."""
    try:
        dpr = res["doc_preprocessor_res"]  # type: ignore[index]
    except (TypeError, KeyError):
        return None
    if isinstance(dpr, list) and dpr:
        dpr = dpr[0]
    if dpr is None:
        return None
    try:
        out = dpr["output_img"]  # type: ignore[index]
    except (TypeError, KeyError):
        return None
    if out is None:
        return None
    arr = out[:, :, ::-1]
    return Image.fromarray(arr)


def page_images_from_vl_results(pages_res: Sequence[Any]) -> List[Image.Image]:
    """Extract one RGB PIL image per PDF page from PaddleOCR-VL ``predict`` results.

    These images come from the pipeline's document preprocessor
    (``doc_preprocessor_res.output_img``). They share the **same pixel grid**
    as ``parsing_res_list`` bounding boxes, which allows:

    - standalone ``PP-DocLayoutV3`` to run on identical rasters; and
    - cropping regions for the per-region VL pass without a second PDF renderer.

    If any page lacks preprocessor output, returns an empty list and logs — the
    caller should treat that as a hard failure for ``paddle_layout_dual_vl``.
    """
    images: List[Image.Image] = []
    for i, res in enumerate(pages_res):
        img = preprocessed_rgb_from_result(res)
        if img is None:
            logger.error(
                "Missing doc_preprocessor_res for page_index=%s; cannot align "
                "PP-DocLayoutV3 with PaddleOCR-VL. Check PDF input and VL "
                "document preprocessing settings.",
                i,
            )
            return []
        images.append(img)
    return images


def _draw_layout_overlay(base: Image.Image, page_regs: List[Dict[str, Any]]) -> Image.Image:
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for reg in page_regs:
        bbox = reg.get("block_bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        rgb = rgb_for_label(str(reg.get("block_label") or "unknown"))
        fill = (*rgb, 50)
        outline = (*rgb, 240)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=3)
        label = str(reg.get("block_label") or "")[:40]
        if not label or font is None:
            continue
        ty = max(0, y1 - 12)
        if hasattr(draw, "textbbox"):
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            tw, th = r - l, b - t
        else:
            tw, th = len(label) * 6, 11
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 2], fill=(*rgb, 230))
        draw.text((x1 + 2, ty + 1), label, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(img, overlay).convert("RGB")


def _truncate_smart(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[: max_chars + 1]
    for sep in (".\n", ". ", "? ", "! "):
        idx = cut.rfind(sep)
        if idx >= max_chars // 2:
            return cut[: idx + 1].strip()
    return cut[:max_chars].rstrip() + "..."


def build_document_summary(
    regions: Sequence[Dict[str, Any]],
    *,
    exclude_labels: Sequence[str],
    max_chars: int,
) -> str:
    ex = {x.lower() for x in exclude_labels}

    lines: List[str] = []
    for r in sort_regions_reading_order(list(regions)):
        lab = str(r.get("block_label") or "").lower()
        if lab in ex:
            continue
        text = str(r.get("block_content") or "").strip()
        if not text:
            continue
        display_lab = str(r.get("block_label") or "")
        lines.append(f"[{display_lab}]\n{text}")
    blob = "\n\n".join(lines)
    return _truncate_smart(blob, max_chars)


def save_layout_artifacts(
    *,
    paper_id: str,
    source_file: str,
    regions: List[Dict[str, Any]],
    pages_res: Sequence[Any],
    layout_dir: Path,
) -> Tuple[str, List[str]]:
    """
    Write ``layout.json`` and per-page overlay PNGs under ``layout_dir/<paper_id>/``.

    Returns repo-relative paths: (layout_json, [overlay_png, ...]).
    """
    paper_layout = layout_dir / paper_id
    paper_layout.mkdir(parents=True, exist_ok=True)

    layout_path = paper_layout / "layout.json"
    doc = {
        "paper_id": paper_id,
        "source_file": source_file,
        "engine": "paddle_vl",
        "regions": regions,
    }
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    overlays: List[str] = []
    by_page: Dict[int, Any] = {}
    for res in pages_res:
        by_page[_page_index_from_result(res)] = res

    for page_idx in sorted(by_page.keys()):
        page_regs = [r for r in regions if int(r.get("page_index", -1)) == page_idx]
        if not page_regs:
            continue
        res = by_page[page_idx]
        base = preprocessed_rgb_from_result(res)
        if base is None:
            logger.warning(
                "Skipping overlay for page %s (no doc preprocessor image). "
                "Layout JSON still lists regions.",
                page_idx,
            )
            continue
        try:
            drawn = _draw_layout_overlay(base, page_regs)
        except Exception as exc:
            logger.warning("Overlay draw failed for page %s: %s", page_idx, exc)
            continue
        out_png = paper_layout / f"page_{page_idx:04d}_overlay.png"
        drawn.save(out_png)
        try:
            overlays.append(str(out_png.relative_to(_REPO_ROOT)))
        except ValueError:
            overlays.append(str(out_png))

    try:
        rel_layout = str(layout_path.relative_to(_REPO_ROOT))
    except ValueError:
        rel_layout = str(layout_path)
    return rel_layout, overlays
