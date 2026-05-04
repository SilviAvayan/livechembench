#!/usr/bin/env bash
# run_pipeline.sh  — Run the full LiveChemBench question-generation pipeline
#
# Prerequisites:
#   export NVIDIA_API_KEY=<your key>
#   micromamba activate chem   (or ensure the chem env python is on PATH)
#
# Usage:
#   bash run_pipeline.sh              # all papers, version 0.3.0
#   bash run_pipeline.sh 0.4.0        # specify a version tag

set -euo pipefail
PYTHON=/Users/savayan/micromamba/envs/chem/bin/python
VERSION="${1:-0.3.0}"

echo "======================================================"
echo "LiveChemBench pipeline — version $VERSION"
echo "======================================================"

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "ERROR: NVIDIA_API_KEY is not set. Export it before running this script."
  exit 1
fi

echo ""
echo "[Step 1/7] Paper quality evaluator"
$PYTHON -m src.agents.paper_quality_evaluator

echo ""
echo "[Step 2/7] Segment selector (novelty bandit)"
$PYTHON -m src.agents.segment_selector

echo ""
echo "[Step 3/7] PubChem linker (entity → CID resolution)"
$PYTHON -m src.agents.pubchem_linker

echo ""
echo "[Step 4/7] Question proposer"
$PYTHON -m src.agents.question_proposer

echo ""
echo "[Step 5/7] Critics (ill-defined + missing conditions)"
$PYTHON -m src.agents.critics

echo ""
echo "[Step 6/7] Question repairer"
$PYTHON -m src.agents.question_repairer

echo ""
echo "[Step 7/7] Novelty selector"
$PYTHON -m src.agents.novelty_selector

echo ""
echo "[Final] Dataset builder → livechembench_v${VERSION}.json"
$PYTHON -m src.agents.dataset_builder --version "$VERSION"

echo ""
echo "======================================================"
echo "Done! Check data/benchmark/livechembench_v${VERSION}.json"
echo "======================================================"
