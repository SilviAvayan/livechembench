from dataclasses import dataclass, field
from typing import List, Optional, Union

@dataclass
class DownloadConfig:
    timeout: int = 30
    retries: int = 3
    concurrent_downloads: int = 5
    chunk_size: int = 1048576

@dataclass
class StorageConfig:
    path: str = "./data"
    max_size: str = "10GB"
    format: str = "json"
    compression: bool = True

@dataclass
class SourceConfig:
    urls: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    rate_limit: int = 100

@dataclass
class AppConfig:
    download: DownloadConfig
    storage: StorageConfig
    source: SourceConfig