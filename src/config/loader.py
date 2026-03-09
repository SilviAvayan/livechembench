print("LOADER FILE EXECUTED")
import yaml
from pydantic import BaseModel
from pathlib import Path

class APIConfig(BaseModel):
    chemrxiv_base_url: str
    user_agent: str

class SearchConfig(BaseModel):
    term: str
    limit: int
    date_range_days: int = 30

class AppConfig(BaseModel):
    api: APIConfig
    search: SearchConfig
    paths: dict[str, str]

def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    print("LOADING CONFIG FROM:", config_path)

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    print("CONFIG CONTENT:", data)

    return AppConfig(**data)

# Singleton instance
config = load_config()






