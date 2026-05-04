#!/usr/bin/env bash
# eval_all_models.sh — Evaluate all available models against LiveChemBench
#
# Usage:
#   bash eval_all_models.sh                          # latest benchmark
#   bash eval_all_models.sh data/benchmark/livechembench_v0.3.0.json
#
# Required env vars:
#   NVIDIA_API_KEY  — for NVIDIA-hosted models
#   OPENAI_API_KEY  — for OpenAI models (optional)
#   ANTHROPIC_API_KEY — for Claude (optional)

set -euo pipefail
PYTHON=/Users/savayan/micromamba/envs/chem/bin/python
NVIDIA_BASE="https://inference-api.nvidia.com/v1"
BENCHMARK="${1:-data/benchmark/livechembench_v0.3.0.json}"

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "ERROR: NVIDIA_API_KEY is not set."
  exit 1
fi

echo "======================================================"
echo "LiveChemBench model evaluation"
echo "Benchmark: $BENCHMARK"
echo "======================================================"

run_nvidia() {
  local model="$1"
  echo ""
  echo "--- Evaluating: $model"
  $PYTHON -m src.agents.model_evaluator \
    --benchmark "$BENCHMARK" \
    --model "$model" \
    --base-url "$NVIDIA_BASE" \
    --api-key-env "NVIDIA_API_KEY"
}

# ---------- NVIDIA-hosted models (not the agent model) ----------
run_nvidia "nvidia/openai/gpt-oss-120b"
run_nvidia "nvidia/qwen/qwen3-next-80b-a3b-instruct"
run_nvidia "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"

# ---------- OpenAI (optional) ----------
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo ""
  echo "--- Evaluating: gpt-4o"
  $PYTHON -m src.agents.model_evaluator \
    --benchmark "$BENCHMARK" \
    --model "gpt-4o" \
    --base-url "https://api.openai.com/v1" \
    --api-key-env "OPENAI_API_KEY"

  echo ""
  echo "--- Evaluating: gpt-4o-mini"
  $PYTHON -m src.agents.model_evaluator \
    --benchmark "$BENCHMARK" \
    --model "gpt-4o-mini" \
    --base-url "https://api.openai.com/v1" \
    --api-key-env "OPENAI_API_KEY"
else
  echo ""
  echo "(Skipping OpenAI models — OPENAI_API_KEY not set)"
fi

# ---------- Anthropic Claude via OpenAI-compat proxy ----------
# Note: Claude doesn't have a native OpenAI-compat /v1 endpoint.
# For Claude, run the questions manually through claude.ai or use the Anthropic SDK.
echo ""
echo "======================================================"
echo "Done. Results in: data/eval_results/"
echo ""
echo "To compare results across models:"
echo "  ls data/eval_results/*.json | xargs -I{} jq '{model: .model, accuracy: .scores.overall, n: .scores.n_total}' {}"
echo "======================================================"
