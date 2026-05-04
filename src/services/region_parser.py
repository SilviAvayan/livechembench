"""
PaddleOCR-VL-1.5 integration for paddle_layout_dual_vl.

whole_paper_pass: one PaddleOCRVL.predict(pdf). Unless
use_pp_doclayout_as_vl_layout_backbone is true, VL does not use PP-DocLayoutV3
as its internal layout; canonical layout boxes come only from standalone
layout_detector.detect_layout_on_page_images. We map parsing_res_list onto
those boxes for content_whole_page.

per_region_pass: for each PP-DocLayoutV3 box, crop the preprocessor page
image and call predict with use_layout_detection=False and a mapped
prompt_label for content_per_region.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config.models import PaddleLayoutDualVLConfig, PpDocLayoutConfig
from src.utils.logger import logger


def create_vl_pipeline(
    dual: PaddleLayoutDualVLConfig, layout: PpDocLayoutConfig
) -> Any:
    """Instantiate ``PaddleOCRVL`` for the whole-paper pass.

    When ``use_pp_doclayout_as_vl_layout_backbone`` is **false** (default),
    we do **not** pass ``layout_detection_model_name=PP-DocLayoutV3`` to VL.
    Canonical layout for segmentation/boxes is **only** from standalone
    PaddleX :func:`paddlex.create_model` (``pp_doclayout`` in config).

    When the flag is **true** (optional "option C" / same-backbone twice),
    VL's internal layout uses the same PP-DocLayoutV3 weights as standalone.
    """
    from paddleocr import PaddleOCRVL

    kwargs: Dict[str, Any] = {
        "pipeline_version": dual.pipeline_version,
        "merge_layout_blocks": dual.whole_page_merge_layout_blocks,
        "use_queues": dual.whole_page_use_queues,
    }
    if dual.device:
        kwargs["device"] = dual.device

    if dual.use_pp_doclayout_as_vl_layout_backbone:
        kwargs["layout_detection_model_name"] = layout.model_name
        if layout.model_dir:
            kwargs["layout_detection_model_dir"] = layout.model_dir
        if layout.threshold is not None:
            kwargs["layout_threshold"] = layout.threshold
        if layout.layout_nms is not None:
            kwargs["layout_nms"] = layout.layout_nms
        if layout.layout_unclip_ratio is not None:
            kwargs["layout_unclip_ratio"] = layout.layout_unclip_ratio
        if layout.layout_merge_bboxes_mode is not None:
            kwargs["layout_merge_bboxes_mode"] = layout.layout_merge_bboxes_mode
    return PaddleOCRVL(**kwargs)


def _paddle_inner_json(res: Any) -> Dict[str, Any]:
    j = getattr(res, "json", None)
    if callable(j):
        j = j()
    if isinstance(j, dict) and "res" in j:
        inner = j.get("res")
        return inner if isinstance(inner, dict) else {}
    return j if isinstance(j, dict) else {}


def _page_index_from_result(res: Any) -> int:
    inner = _paddle_inner_json(res)
    v = inner.get("page_index")
    if isinstance(v, list) and v:
        v = v[0]
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0


def _bbox_int(box: Sequence[Any]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(v))) for v in box[:4])  # type: ignore[return-value]


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def whole_paper_pass(
    *,
    pdf_path: Path,
    pipeline: Any,
    dual: PaddleLayoutDualVLConfig,
) -> Tuple[List[Any], List[Any]]:
    """Run the whole-paper VL pass and return ``(pages_res, restructured_res)``.

    ``restructured_res`` is the post-``restructure_pages`` view used for table
    merging / multi-level title relevel; ``pages_res`` is the raw per-page
    output used for layout-aligned content lookup.
    """
    pages_res = list(pipeline.predict(input=str(pdf_path)))
    if not pages_res:
        return [], []
    restructured = pipeline.restructure_pages(
        pages_res,
        merge_tables=dual.merge_tables,
        relevel_titles=dual.relevel_titles,
        concatenate_pages=dual.concatenate_pages,
    )
    return pages_res, list(restructured) if restructured else []


def index_whole_paper_blocks_by_bbox(
    pages_res: Sequence[Any],
) -> Dict[int, List[Tuple[Tuple[int, int, int, int], str, str]]]:
    """Build ``{page_index: [(bbox, label, content), ...]}`` from VL results."""
    out: Dict[int, List[Tuple[Tuple[int, int, int, int], str, str]]] = {}
    for res in pages_res:
        page_idx = _page_index_from_result(res)
        inner = _paddle_inner_json(res)
        bucket = out.setdefault(page_idx, [])
        for block in inner.get("parsing_res_list") or []:
            if not isinstance(block, dict):
                continue
            bbox = block.get("block_bbox") or block.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            label = str(block.get("block_label") or block.get("label") or "")
            content = str(block.get("block_content") or block.get("content") or "")
            bucket.append((_bbox_int(bbox), label, content))
    return out


def attach_whole_page_content(
    regions: List[Dict[str, Any]],
    *,
    whole_page_blocks: Dict[
        int, List[Tuple[Tuple[int, int, int, int], str, str]]
    ],
    iou_match_threshold: float = 0.5,
) -> None:
    """In-place: set ``content_whole_page`` on each region by bbox alignment.

    Tries exact bbox match first (the model is the same in both runs), then
    falls back to highest-IoU match within the page. Regions with no good
    match get an empty string.
    """
    for r in regions:
        page_idx = int(r.get("page_index", 0))
        target = _bbox_int(r.get("block_bbox") or [0, 0, 0, 0])
        candidates = whole_page_blocks.get(page_idx) or []

        match_content = ""
        match_label = ""
        for bbox, label, content in candidates:
            if bbox == target:
                match_content = content
                match_label = label
                break

        if not match_content and candidates:
            best_iou = 0.0
            for bbox, label, content in candidates:
                iou = _bbox_iou(bbox, target)
                if iou > best_iou:
                    best_iou = iou
                    match_content = content
                    match_label = label
            if best_iou < iou_match_threshold:
                match_content = ""
                match_label = ""

        r["content_whole_page"] = match_content
        if match_label:
            r["whole_page_block_label"] = match_label


def _crop_region(page_image: Any, bbox: Sequence[int], pad: int) -> Any:
    x1, y1, x2, y2 = bbox[:4]
    w, h = page_image.width, page_image.height
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return page_image.crop((x1, y1, x2, y2))


def _prompt_label_for(
    block_label: str, dual: PaddleLayoutDualVLConfig
) -> Optional[str]:
    if block_label.lower() in {x.lower() for x in dual.per_region_skip_labels}:
        return None
    return dual.per_region_label_map.get(
        block_label.lower(), dual.per_region_default_prompt_label
    )


def _per_region_predict(
    pipeline: Any, crop: Any, prompt_label: str
) -> str:
    """Run a single VL call on a cropped region and return its content text."""
    try:
        results = list(
            pipeline.predict(
                input=crop,
                use_layout_detection=False,
                prompt_label=prompt_label,
            )
        )
    except Exception as exc:
        logger.warning(
            "Per-region VL call failed (prompt_label=%s): %s",
            prompt_label,
            exc,
        )
        return ""

    if not results:
        return ""

    inner = _paddle_inner_json(results[0])
    blocks = inner.get("parsing_res_list") or []
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        c = b.get("block_content") or b.get("content")
        if c:
            parts.append(str(c))
    return "\n\n".join(parts).strip()


def per_region_pass(
    regions: List[Dict[str, Any]],
    *,
    page_images: Sequence[Any],
    pipeline: Any,
    dual: PaddleLayoutDualVLConfig,
) -> None:
    """In-place: fill ``content_per_region`` and ``prompt_label`` per region."""
    if not dual.enable_per_region_pass:
        for r in regions:
            r.setdefault("content_per_region", "")
            r.setdefault("prompt_label", "")
        return

    pad = max(0, int(dual.per_region_crop_padding_px))

    def _job(idx: int) -> Tuple[int, str, str]:
        r = regions[idx]
        page_idx = int(r.get("page_index", 0))
        if page_idx >= len(page_images):
            return idx, "", ""
        prompt = _prompt_label_for(str(r.get("block_label") or ""), dual)
        if prompt is None:
            return idx, "", ""
        crop = _crop_region(page_images[page_idx], r.get("block_bbox") or [], pad)
        if crop is None:
            return idx, "", prompt
        return idx, _per_region_predict(pipeline, crop, prompt), prompt

    workers = max(1, int(dual.per_region_max_concurrency))
    if workers == 1:
        for i in range(len(regions)):
            _, content, prompt = _job(i)
            regions[i]["content_per_region"] = content
            regions[i]["prompt_label"] = prompt
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, content, prompt in pool.map(_job, range(len(regions))):
            regions[idx]["content_per_region"] = content
            regions[idx]["prompt_label"] = prompt
