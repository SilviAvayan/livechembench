# Contributing

This project lives at **https://github.com/SilviAvayan/livechembench**.

## Workflow

1. Fork or get write access to the repo.
2. Create a branch from `main`.
3. Open a **pull request** into `main` when ready.
4. CI runs on pushes and PRs: config validation and lightweight Python imports (no Paddle stack in CI).

## Local setup

```bash
git clone https://github.com/SilviAvayan/livechembench.git
cd livechembench
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Segmentation with Paddle engines requires [PaddlePaddle](https://www.paddlepaddle.org.cn/) and:

```bash
pip install -r requirements-paddleocr-vl.txt
```

## Secrets

Do **not** commit API keys. Prefer `NCBI_API_KEY` in the environment (see `src/config/loader.py`); override values in `config.yaml` only for local use and keep keys out of git history.

## Checks before you push

```bash
python -c "import yaml; from pathlib import Path; import sys; sys.path.insert(0,'.'); from src.config.models import AppConfig; AppConfig.model_validate(yaml.safe_load(Path('config.yaml').read_text()))"
```
