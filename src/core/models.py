from pydantic import BaseModel
from typing import Dict

class APIConfig(BaseModel):
    chemrxiv_base_url: str
    user_agent: str


class SearchConfig(BaseModel):
    term: str
    limit: int
    date_range_days: int


class AppConfig(BaseModel):
    api: APIConfig
    search: SearchConfig
    paths: Dict[str, str]