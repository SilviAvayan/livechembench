from pydantic import BaseModel, Field, ConfigDict
from typing import Dict

class APIConfig(BaseModel):
    """Configuration for external API endpoints and identity."""
    chemrxiv_base_url: str = Field(..., description="The base URL for the ChemRxiv/Figshare API")
    user_agent: str = Field(..., description="The User-Agent string for HTTP requests")

class SearchConfig(BaseModel):
    """Configuration for search parameters and filters."""
    term: str = Field(..., description="The search term used to filter papers")
    limit: int = Field(default=100, description="Number of results per API page")
    date_range_days: int = Field(..., description="Number of days to look back for new papers")


class PathConfig(BaseModel):
    """Configuration for local directory paths."""
    raw_papers: str
    benchmark: str = "data/benchmark" # You can add defaults for others in your tree
    segments: str = "data/segments"

class AppConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    api: APIConfig
    search: SearchConfig
    paths: PathConfig