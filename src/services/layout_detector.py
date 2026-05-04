"""
Stage 1 of the ``paddle_layout_dual_vl`` pipeline: standalone PP-DocLayoutV3.

Runs PaddleX's ``PP-DocLayoutV3`` model on already-rendered RGB page images
(provided by the whole-paper VL pass via its ``doc_preprocessor_res``) and
returns a list of region dicts in the same shape produced by the legacy
paddle_vl engine, so the existing overlay/serialization helpers in
``layout_export.py`` keep working unchanged:

    {
      "page_index": int,
      "block_label": str,        # one of PP-DocLayoutV3's 25 classes
      "block_bbox": [int, int, int, int],   # [x1, y1, x2, y2]
      "block_content": "",       # always empty here (layout is text-free)
      "score": float,            # detection confidence (extra field)
      "block_id": int,           # reading-order index within the document
      "page_image_size": [w, h], # pixel size of the rendered page image
    }

By reusing VL's internal renders we avoid pulling in a second PDF renderer
(PyMuPDF) and we guarantee that bboxes from PP-DocLayoutV3 line up exactly
with the page coordinate system VL uses for its own ``parsing_res_list``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.models import PpDocLayoutConfig
from src.utils.logger import logger


def create_pp_doclayout_model(cfg: PpDocLayoutConfig) -> Any:
    """Instantiate the standalone PP-DocLayoutV3 PaddleX model once per run."""
    from paddlex import create_model

    kwargs: Dict[str, Any] = {"model_name": cfg.model_name}
    if cfg.model_dir:
        kwargs["model_dir"] = cfg.model_dir
    if cfg.device:
        kwargs["device"] = cfg.device
    if cfg.threshold is not None:
        kwargs["threshold"] = cfg.threshold
    if cfg.layout_nms is not None:
        kwargs["layout_nms"] = cfg.layout_nms
    if cfg.layout_unclip_ratio is not None:
        kwargs["layout_unclip_ratio"] = cfg.layout_unclip_ratio
    if cfg.layout_merge_bboxes_mode is not None:
        kwargs["layout_merge_bboxes_mode"] = cfg.layout_merge_bboxes_mode
    return create_model(**kwargs)


def _result_to_box_list(res: Any) -> List[Dict[str, Any]]:
    """Extract the ``boxes`` list from a single PaddleX layout result."""
    j = getattr(res, "json", None)
    if callable(j):
        j = j()
    if isinstance(j, dict) and "res" in j and isinstance(j["res"], dict):
        j = j["res"]
    if not isinstance(j, dict):
        return []
    boxes = j.get("boxes")
    if not isinstance(boxes, list):
        return []
    return [b for b in boxes if isinstance(b, dict)]


def _box_to_region(
    box: Dict[str, Any],
    *,
    page_index: int,
    page_size: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    coord = box.get("coordinate") or box.get("bbox")
    label = box.get("label")
    if coord is None or label is None or len(coord) < 4:
        return None
    x1, y1, x2, y2 = (int(round(float(v))) for v in coord[:4])
    score = box.get("score")
    order = box.get("order")
    return {
        "page_index": page_index,
        "block_label": str(label),
        "block_bbox": [x1, y1, x2, y2],
        "block_content": "",
        "score": float(score) if score is not None else None,
        "page_order": int(order) if order is not None else None,
        "page_image_size": [int(page_size[0]), int(page_size[1])],
    }


def detect_layout_on_page_images(
    page_images: Sequence[Any],
    cfg: PpDocLayoutConfig,
    *,
    model: Any = None,
) -> List[Dict[str, Any]]:
    """Run PP-DocLayoutV3 over already-rendered page images.

    ``page_images`` must be the RGB PIL images emitted by the PaddleOCR-VL
    whole-paper pass via ``doc_preprocessor_res.output_img``. Using those
    exact images is important: it keeps PP-DocLayoutV3 bboxes in the same
    pixel coordinate system as VL's ``parsing_res_list``.
    """
    if not page_images:
        return []

    if model is None:
        model = create_pp_doclayout_model(cfg)

    import numpy as np

    page_arrays = [np.asarray(img) for img in page_images]
    results = list(model.predict(page_arrays, batch_size=1))

    if len(results) != len(page_images):
        logger.warning(
            "PP-DocLayoutV3 returned %d results for %d page images; "
            "regions may be misaligned.",
            len(results),
            len(page_images),
        )

    regions: List[Dict[str, Any]] = []
    for page_idx, (img, res) in enumerate(zip(page_images, results)):
        page_size = (img.width, img.height)
        for box in _result_to_box_list(res):
            row = _box_to_region(box, page_index=page_idx, page_size=page_size)
            if row is not None:
                regions.append(row)

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
        bbox = r.get("block_bbox") or [0, 0, 0, 0]
        order = r.get("page_order")
        order_key = order if isinstance(order, int) else 1_000_000
        return (
            int(r.get("page_index", 0)),
            order_key,
            int(bbox[1]) if len(bbox) > 1 else 0,
            int(bbox[0]) if bbox else 0,
        )

    regions.sort(key=_sort_key)
    for global_id, r in enumerate(regions):
        r["block_id"] = global_id

    return regions


def write_layout_artifact(
    *,
    paper_id: str,
    source_file: str,
    regions: Sequence[Dict[str, Any]],
    layout_dir: Path,
) -> Path:
    """Persist the Stage-1 layout JSON at ``layout_dir/<paper_id>/layout.json``."""
    import json

    paper_layout = layout_dir / paper_id
    paper_layout.mkdir(parents=True, exist_ok=True)
    layout_path = paper_layout / "layout.json"
    doc = {
        "paper_id": paper_id,
        "source_file": source_file,
        "engine": "paddle_layout_dual_vl",
        "stage": "pp_doclayoutv3",
        "regions": list(regions),
    }
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return layout_path


def write_layout_overlays(
    *,
    paper_id: str,
    regions: Sequence[Dict[str, Any]],
    page_images: Sequence[Any],
    layout_dir: Path,
) -> List[Path]:
    """Write per-page colored overlay PNGs from rendered images + Stage-1 boxes.

    Reuses the drawing helper from ``layout_export`` so colors stay consistent
    with overlays produced by the legacy paddle_vl engine.
    """
    from src.services.layout_export import _draw_layout_overlay  # type: ignore

    paper_layout = layout_dir / paper_id
    paper_layout.mkdir(parents=True, exist_ok=True)

    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for r in regions:
        by_page.setdefault(int(r.get("page_index", 0)), []).append(dict(r))

    overlay_paths: List[Path] = []
    for page_idx, base in enumerate(page_images):
        page_regs = by_page.get(page_idx) or []
        if not page_regs:
            continue
        try:
            drawn = _draw_layout_overlay(base, page_regs)
        except Exception as exc:
            logger.warning("Overlay draw failed for page %s: %s", page_idx, exc)
            continue
        out_png = paper_layout / f"page_{page_idx:04d}_overlay.png"
        try:
            drawn.save(out_png)
            overlay_paths.append(out_png)
        except Exception as exc:
            logger.warning("Could not save overlay %s: %s", out_png, exc)

    return overlay_paths
