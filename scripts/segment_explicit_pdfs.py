#!/usr/bin/env python3
"""Run paddle_layout_dual_vl on a fixed list of PDFs (under data/raw_papers/).

Loads PaddleOCR-VL and PP-DocLayoutV3 once, then segments each file.
Usage:
  python scripts/segment_explicit_pdfs.py pmc_12987937.pdf pmc_12987760.pdf
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config.loader import config
from src.services.segment_pipeline import SegmentPipeline
from src.services.segmenter import segment_paper


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF filenames (must exist in data/raw_papers/ or pass --raw-dir)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override raw papers directory (default: config paths.raw_papers)",
    )
    args = parser.parse_args()

    sp = SegmentPipeline()
    raw = args.raw_dir or sp.raw_path
    if not raw.is_absolute():
        raw = _REPO / raw
    raw = raw.resolve()

    eng = config.segmentation.engine
    if eng != "paddle_layout_dual_vl":
        print(
            f"Warning: config has engine={eng!r}; this script is intended for "
            f"paddle_layout_dual_vl. Set segmentation.engine in config.yaml.",
            file=sys.stderr,
        )

    paddle_pipeline = None
    layout_model = None
    if eng in ("paddle_layout_dual_vl", "paddle_vl"):
        if eng == "paddle_layout_dual_vl":
            paddle_pipeline = sp._load_dual_vl_pipeline()
            layout_model = sp._load_layout_model()
        else:
            paddle_pipeline = sp._load_paddle_vl_pipeline()

    for name in args.pdfs:
        pdf = (raw / name).resolve() if not Path(name).is_absolute() else Path(name)
        if not pdf.is_file():
            print(f"SKIP (not found): {pdf}", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        print(f"=== {pdf.name} ===", flush=True)
        paper = segment_paper(
            pdf,
            sp.cfg,
            sp.assets_root,
            layout_root=sp.layout_root,
            paddle_pipeline=paddle_pipeline,
            layout_model=layout_model,
        )
        sp._save(paper)
        dt = time.perf_counter() - t0
        out = sp._output_path(paper.paper_id)
        print(
            f"  status={paper.extraction_status} time={dt:.1f}s -> {out}",
            flush=True,
        )
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
