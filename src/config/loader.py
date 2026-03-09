# src/config/loader.py
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# Use relative import to avoid circular imports
from .models import AppConfig, DownloadConfig, StorageConfig, SourceConfig


class ConfigLoader:
    _instance: Optional[AppConfig] = None

    @classmethod
    def load(cls, config_path: str | Path) -> AppConfig:
        """Load configuration from YAML file."""
        if cls._instance is not None:
            return cls._instance

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        # Validate required sections
        cls._validate_config(data)

        config = AppConfig(
            download=DownloadConfig(**data.get("download", {})),
            storage=StorageConfig(**data.get("storage", {})),
            source=SourceConfig(**data.get("source", {})),
        )

        cls._instance = config
        return config

    @classmethod
    def _validate_config(cls, data: Dict[str, Any]) -> None:
        """Validate that required config sections exist."""
        required_sections = ["download", "storage", "source"]
        missing = [section for section in required_sections if section not in data]
        
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None