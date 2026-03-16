import os
from pathlib import Path

import yaml

from src.config.models import AppConfig  # Import the consolidated model


def load_config() -> AppConfig:
    """Load application configuration from YAML + environment overrides."""
    # Go up 3 levels from src/config/loader.py to find config.yaml
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Allow sensitive values to be injected via environment variables
    env_ncbi = os.getenv("NCBI_API_KEY")
    if env_ncbi:
        data.setdefault("api", {})["ncbi_api_key"] = env_ncbi

    # In Pydantic V2, this converts the dict (and nested dicts) into models
    return AppConfig.model_validate(data)


# Singleton instance
config = load_config()