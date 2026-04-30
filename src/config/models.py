from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class APIConfig(BaseModel):
    chemrxiv_base_url: str
    user_agent: str
    ncbi_api_key: Optional[str] = None


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


class NoveltyConfig(BaseModel):
    top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 1024


class EntityExtractorConfig(BaseModel):
    temperature: float = 0.2
    max_tokens: int = 2048
    pubchem_timeout: int = 10
    max_entities_per_paper: int = 20


class QuestionProposerConfig(BaseModel):
    questions_per_paper: int = 5
    question_types: List[str] = Field(
        default=["multiple_choice", "free_text", "numerical"]
    )
    temperature: float = 0.7
    max_tokens: int = 4096


class TriCriticConfig(BaseModel):
    max_iterations: int = 3
    temperature: float = 0.4
    max_tokens: int = 2048


class BaselineModel(BaseModel):
    name: str
    base_url: str


class EvaluatorConfig(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 512
    baseline_models: List[BaselineModel] = Field(default_factory=list)


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    primary_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    novelty_selector: NoveltyConfig = Field(default_factory=NoveltyConfig)
    entity_extractor: EntityExtractorConfig = Field(default_factory=EntityExtractorConfig)
    question_proposer: QuestionProposerConfig = Field(default_factory=QuestionProposerConfig)
    tri_critic: TriCriticConfig = Field(default_factory=TriCriticConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)


class OutputConfig(BaseModel):
    dataset_jsonl: str = "data/output/dataset.jsonl"
    provenance_jsonl: str = "data/output/provenance.jsonl"
    verifier_configs: str = "data/output/verifier_configs.json"
    leaderboard_json: str = "data/output/leaderboard.json"
    leaderboard_md: str = "data/output/leaderboard.md"


class AppConfig(BaseModel):
    # This tells Pydantic how to handle the dict from yaml
    model_config = ConfigDict(extra='ignore')

    api: APIConfig
    search: SearchConfig
    paths: PathConfig
    segmentation: SegmentationConfig = SegmentationConfig()
    agents: AgentConfig = Field(default_factory=AgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)