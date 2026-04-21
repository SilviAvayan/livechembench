from typing import List, Literal, Optional
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


class SegmentationConfig(BaseModel):
    """Configuration for the paper segmentation pipeline."""
    engine: Literal["paddle_vl", "docling"] = "paddle_vl"
    paddle_vl: PaddleVLSegmentationConfig = Field(
        default_factory=PaddleVLSegmentationConfig
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


class AppConfig(BaseModel):
    # This tells Pydantic how to handle the dict from yaml
    model_config = ConfigDict(extra='ignore')

    api: APIConfig
    search: SearchConfig
    paths: PathConfig
    segmentation: SegmentationConfig = SegmentationConfig()