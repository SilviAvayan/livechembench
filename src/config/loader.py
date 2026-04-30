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

    env_nvidia = os.getenv("NVIDIA_API_KEY")
    if env_nvidia:
        data.setdefault("agents", {})["api_key"] = env_nvidia
    else:
        # Expand ${NVIDIA_API_KEY} placeholder from YAML if env var not set
        raw_key = data.get("agents", {}).get("api_key", "")
        if raw_key.startswith("${") and raw_key.endswith("}"):
            var_name = raw_key[2:-1]
            data.setdefault("agents", {})["api_key"] = os.getenv(var_name, "")

    # In Pydantic V2, this converts the dict (and nested dicts) into models
    return AppConfig.model_validate(data)


# Singleton instance
config = load_config()