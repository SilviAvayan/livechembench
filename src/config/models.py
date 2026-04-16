from typing import List
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


class SegmentationConfig(BaseModel):
    """Configuration for the paper segmentation pipeline."""
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