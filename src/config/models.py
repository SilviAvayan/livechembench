from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class APIConfig(BaseModel):
    chemrxiv_base_url: str
    user_agent: str
    ncbi_api_key: str | None = None


class SearchConfig(BaseModel):
    term: str
    limit: int
    date_range_days: int


class PathConfig(BaseModel):
    raw_papers: str
    segmented_papers: str = "data/segmented_papers"
    # Per-paper exports (figure crops, optional sidecar files) live under this tree.
    segmented_assets: str = "data/segmented_papers/assets"
    # Per-paper layout JSON (bbox/label/text) and colored overlay PNGs (PaddleOCR-VL).
    segmented_layout: str = "data/segmented_papers/layout"


class PaddleVLSegmentationConfig(BaseModel):
    """PaddleOCR-VL-1.5 doc-parser pipeline (see PaddleOCR docs: PaddleOCR-VL)."""

    pipeline_version: str = "v1.5"
    device: Optional[str] = None
    merge_tables: bool = True
    relevel_titles: bool = True
    concatenate_pages: bool = True
    table_labels: List[str] = Field(default_factory=lambda: ["table"])
    figure_labels: List[str] = Field(
        default_factory=lambda: ["image", "chart", "figure"]
    )
    document_summary_max_chars: int = 2500
    summary_exclude_labels: List[str] = Field(
        default_factory=lambda: [
            "header",
            "footer",
            "header_image",
            "footer_image",
            "footnote",
            "page_number",
            "aside_text",
        ]
    )


class PpDocLayoutConfig(BaseModel):
    """Standalone PP-DocLayoutV3 layout detector (PaddleX module)."""

    model_name: str = "PP-DocLayoutV3"
    model_dir: Optional[str] = None
    device: Optional[str] = None
    threshold: Optional[float] = None
    layout_nms: Optional[bool] = None
    layout_unclip_ratio: Optional[float] = None
    # 'large' | 'small' | 'union' (None => model default)
    layout_merge_bboxes_mode: Optional[str] = None
    # Kept for forward compatibility; ``paddle_layout_dual_vl`` does not use a
    # second PDF renderer — it reuses PaddleOCR-VL's document-preprocessor page
    # images for PP-DocLayoutV3 and per-region crops.
    render_dpi: int = 200


class PaddleLayoutDualVLConfig(BaseModel):
    """Explicit dual-parse pipeline: standalone PP-DocLayoutV3 + PaddleOCR-VL-1.5.

    Execution order in code: (1) one ``PaddleOCRVL.predict(pdf)`` pass for
    whole-document parsing and preprocessor page images (layout inside VL uses
    the **default** detector unless ``use_pp_doclayout_as_vl_layout_backbone``);
    (2) standalone ``PP-DocLayoutV3`` for **canonical** layout boxes; (3a) join
    VL ``parsing_res_list`` onto those boxes → ``content_whole_page``; (3b) VL
    per crop → ``content_per_region``.
    """

    # VL whole-paper pass: pipeline version and device (layout weights for VL are
    # **not** tied to ``pp_doclayout`` unless use_pp_doclayout_as_vl_layout_backbone).
    pipeline_version: str = "v1.5"
    device: Optional[str] = None

    # If true (optional "option C" experiment), PaddleOCR-VL's whole-paper pass also
    # uses ``PP-DocLayoutV3`` as its *internal* layout backbone — the same
    # checkpoint as standalone layout, so layout runs twice. Default **false**:
    # layout for your paper is **only** the standalone PP-DocLayoutV3 run; the
    # whole-paper VL pass uses the pipeline default layout detector for parsing
    # only, and we match its ``parsing_res_list`` onto PP-DocLayoutV3 boxes by bbox.
    use_pp_doclayout_as_vl_layout_backbone: bool = False

    # Restructure flags applied after the whole-paper VL pass.
    merge_tables: bool = True
    relevel_titles: bool = True
    concatenate_pages: bool = True

    # Whole-paper VL: merge_layout_blocks. When
    # ``use_pp_doclayout_as_vl_layout_backbone`` is true, set false to align
    # VL internal boxes with standalone PP-DocLayoutV3; when false, this only
    # affects VL's default layout merge behavior.
    whole_page_merge_layout_blocks: bool = False
    whole_page_use_queues: bool = True

    # Per-region VL pass options.
    # If false, skip Stage 2b entirely (region.content_per_region stays empty).
    enable_per_region_pass: bool = True
    # Pad each crop by this many pixels on every side (helps VL with edge tokens).
    per_region_crop_padding_px: int = 8
    # Max parallel per-region VL calls (only effective with a vLLM/SGLang server).
    per_region_max_concurrency: int = 8
    # PP-DocLayoutV3 labels to skip entirely in Stage 2b (e.g. pure-image regions
    # where VL parsing of the crop is wasteful).
    per_region_skip_labels: List[str] = Field(
        default_factory=lambda: [
            "header_image",
            "footer_image",
            "seal",
            "number",
            "page_number",
        ]
    )

    # PP-DocLayoutV3 → VL prompt_label mapping. Keys are PP-DocLayoutV3 labels
    # (lowercased); values are VL prompt labels in {"ocr","formula","table","chart"}.
    # Anything not in this map falls back to ``per_region_default_prompt_label``.
    per_region_label_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "table": "table",
            "table_title": "ocr",
            "chart": "chart",
            "chart_title": "ocr",
            "formula": "formula",
            "display_formula": "formula",
            "inline_formula": "formula",
            "image": "ocr",
            "figure_title": "ocr",
        }
    )
    per_region_default_prompt_label: str = "ocr"

    # Same summary configuration as the legacy paddle_vl engine.
    table_labels: List[str] = Field(default_factory=lambda: ["table"])
    figure_labels: List[str] = Field(
        default_factory=lambda: ["image", "chart", "figure"]
    )
    document_summary_max_chars: int = 2500
    summary_exclude_labels: List[str] = Field(
        default_factory=lambda: [
            "header",
            "footer",
            "header_image",
            "footer_image",
            "footnote",
            "page_number",
            "aside_text",
        ]
    )


class SegmentationConfig(BaseModel):
    """Configuration for the paper segmentation pipeline."""
    engine: Literal["paddle_layout_dual_vl", "paddle_vl", "docling"] = (
        "paddle_layout_dual_vl"
    )
    paddle_vl: PaddleVLSegmentationConfig = Field(
        default_factory=PaddleVLSegmentationConfig
    )
    pp_doclayout: PpDocLayoutConfig = Field(default_factory=PpDocLayoutConfig)
    paddle_layout_dual_vl: PaddleLayoutDualVLConfig = Field(
        default_factory=PaddleLayoutDualVLConfig
    )
    max_key_points: int = 5
    min_section_chars: int = 50
    abstract_headings: List[str] = Field(
        default=["abstract", "summary", "overview"]
    )
    conclusion_headings: List[str] = Field(
        default=[
            "conclusion",
            "conclusions",
            "concluding remarks",
            "summary and conclusion",
            "final remarks",
            "summary and outlook",
            "outlook",
        ]
    )


class OrchestrationConfig(BaseModel):
    """Agent/workflow runner defaults (see ``src.agent``)."""

    stop_on_first_failure: bool = True
    persist_run_manifests: bool = True
    runs_directory: str = "data/runs"


class AppConfig(BaseModel):
    # This tells Pydantic how to handle the dict from yaml
    model_config = ConfigDict(extra='ignore')

    api: APIConfig
    search: SearchConfig
    paths: PathConfig
    segmentation: SegmentationConfig = SegmentationConfig()
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)