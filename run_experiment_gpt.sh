#!/usr/bin/env bash
# run_experiment_gpt.sh — Proposer-bias experiment: GPT-oss-120b vs Gemini
#
# Generates benchmark v0.6.0 using nvidia/openai/gpt-oss-120b as the question
# proposer (instead of Gemini) on the same 4 papers that contributed to v0.3.0.
# All intermediate data lands in data/experiment/ so the main pipeline is untouched.
#
# Usage:
#   export NVIDIA_API_KEY=<key>
#   bash run_experiment_gpt.sh
#
# Outputs:
#   data/benchmark/livechembench_v0.6.0.json  — GPT-proposed questions
#   data/eval_results/v0.6.0_*.json           — per-model eval results
#   (v0.3.0 eval results already exist for comparison)

set -uo pipefail
PYTHON=/Users/savayan/micromamba/envs/chem/bin/python
EXP=data/experiment

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "ERROR: NVIDIA_API_KEY not set."
  exit 1
fi

mkdir -p "$EXP/proposed" "$EXP/critiques" "$EXP/repaired" "$EXP/selected"

run_step() {
  local name="$1"; shift
  echo ""
  echo "[$(date +%H:%M:%S)] $name"
  if "$@"; then
    echo "[$(date +%H:%M:%S)] $name: OK"
  else
    echo "[$(date +%H:%M:%S)] $name: FAILED (exit=$?)"
  fi
}

# The 4 papers that contributed questions to v0.3.0
PAPERS=(pmc_12987923 pmc_12987937 pubmed_12478176 pubmed_12987816)

echo "========================================================"
echo "Proposer-bias experiment — GPT-oss-120b proposer"
echo "Papers: ${PAPERS[*]}"
echo "Started: $(date)"
echo "========================================================"

# Step 1: Propose questions with GPT proposer
for pid in "${PAPERS[@]}"; do
  run_step "Proposer — $pid" \
    $PYTHON -m src.agents.question_proposer \
      --paper-id "$pid" \
      --output-dir "$EXP/proposed"
done

# Step 2: Critics
run_step "Critics" \
  $PYTHON -m src.agents.critics \
    --proposed-dir "$EXP/proposed" \
    --output-dir   "$EXP/critiques"

# Step 3: Repairer
run_step "Question repairer" \
  $PYTHON -m src.agents.question_repairer \
    --proposed-dir  "$EXP/proposed" \
    --critiques-dir "$EXP/critiques" \
    --output-dir    "$EXP/repaired"

# Step 4: Novelty selector
run_step "Novelty selector" \
  $PYTHON -m src.agents.novelty_selector \
    --repaired-dir "$EXP/repaired" \
    --output-dir   "$EXP/selected"

# Step 5: Dataset builder → v0.6.0
run_step "Dataset builder → v0.6.0" \
  $PYTHON -m src.agents.dataset_builder \
    --version    "0.6.0" \
    --selected-dir  "$EXP/selected" \
    --critiques-dir "$EXP/critiques"

# Step 6: Evaluate same models on v0.6.0 (GPT-generated)
echo ""
echo "[$(date +%H:%M:%S)] Evaluating models on v0.6.0 (GPT-proposed) ..."

BENCHMARK=data/benchmark/livechembench_v0.6.0.json
if [ ! -f "$BENCHMARK" ]; then
  echo "ERROR: $BENCHMARK not found — dataset_builder must have failed."
  exit 1
fi

for MODEL in \
  "nvidia/openai/gpt-oss-120b" \
  "nvidia/nvidia/Nemotron-3-Nano-30B-A3B" \
  "nvidia/qwen/qwen3-next-80b-a3b-instruct" \
  "gcp/google/gemini-3.1-flash-lite-preview"
do
  run_step "Eval — $MODEL on v0.6.0" \
    $PYTHON -m src.agents.model_evaluator \
      --model "$MODEL" \
      --benchmark "$BENCHMARK"
done

# Step 7: Also run Gemini Flash-Lite on v0.3.0 if not already done
V03=data/benchmark/livechembench_v0.3.0.json
V03_GEMINI=data/eval_results/v0.3.0_gcp_google_gemini-3.1-flash-lite-preview.json
if [ -f "$V03" ] && [ ! -f "$V03_GEMINI" ]; then
  run_step "Eval — Gemini Flash-Lite on v0.3.0 (for comparison)" \
    $PYTHON -m src.agents.model_evaluator \
      --model "gcp/google/gemini-3.1-flash-lite-preview" \
      --benchmark "$V03"
fi

echo ""
echo "========================================================"
echo "Done at $(date)"
echo ""
echo "Results:"
echo "  v0.3.0 (Gemini-proposed):  data/eval_results/v0.3.0_*.json"
echo "  v0.6.0 (GPT-proposed):     data/eval_results/v0.6.0_*.json"
echo ""
echo "Compare with:"
echo "  python experiments/compare_proposer_bias.py"
echo "========================================================"
