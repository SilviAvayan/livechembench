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

class AppConfig(BaseModel):
    # This tells Pydantic how to handle the dict from yaml
    model_config = ConfigDict(extra='ignore')
    
    api: APIConfig
    search: SearchConfig
    paths: PathConfig