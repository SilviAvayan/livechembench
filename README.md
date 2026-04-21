# LiveChemBench

A live, monthly-updated chemistry benchmark built from recent papers with PubChem-grounded verification.

Capstone project.

## Pipeline

1. **Download** — Fetch recent papers (PubMed, ChemRxiv) into `data/raw_papers/`.
2. **Segment** — Parse PDFs and extract abstract, main points, conclusion, table text, and figure crops.

Segmentation uses **PaddleOCR-VL-1.5** by default (`segmentation.engine: paddle_vl` in `config.yaml`): full-document markdown via `restructure_pages`, heuristic sections from headings, `parsing_res_list` blocks labeled `table` for tables, and images from the pipeline `markdown` output saved under `data/segmented_papers/assets/<paper_id>/`.

Optional **Docling** backend: set `segmentation.engine: docling` (no table list or figure exports).

### Running

```bash
# Fetch papers (default)
python -m src.main

# Segment all PDFs in data/raw_papers/
python -m src.main --segment

# Smoke test: only the first 5 PDFs (sorted by filename)
python -m src.main --segment --limit 5
```

PaddleOCR-VL requires extra dependencies; after installing [PaddlePaddle](https://www.paddlepaddle.org.cn/) for your OS/GPU, install:

```bash
pip install -r requirements-paddleocr-vl.txt
```