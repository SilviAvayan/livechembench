# LiveChemBench

[![CI](https://github.com/SilviAvayan/livechembench/actions/workflows/ci.yml/badge.svg)](https://github.com/SilviAvayan/livechembench/actions/workflows/ci.yml)

A dynamic, contamination-resistant chemistry reasoning benchmark built automatically from recent peer-reviewed literature. The pipeline ingests raw chemistry PDFs, generates structured benchmark questions, enforces quality through a multi-agent critic loop, verifies answers deterministically using RDKit and PubChem, and evaluates LLMs against the resulting dataset — all without human annotation.

## Motivation

Static chemistry benchmarks become unreliable over time as LLMs train on publicly available test questions. LiveChemBench addresses this by generating fresh questions from newly published papers on demand, making contamination structurally difficult.

## Pipeline

```
PDFs
 └─► Segmentation (PaddleOCR-VL)
      └─► A1 Segment Selector     novelty-bandit scores segments (UCB policy)
           └─► Quality Evaluator  filters for real research papers
                └─► A3 PubChem Linker   resolves entities → CIDs + properties
                     └─► A4 Question Proposer   generates T1/T2/T3 questions
                          └─► A5 Tri-Critic (Critics 1+2+3)
                          │     ├─ Critic 1: ill-defined (dual-solver check)
                          │     ├─ Critic 2: missing conditions (structured patch list)
                          │     └─ Critic 3: blind-solver (guessable from PubChem?)
                               └─► A6 Question Repairer   LLM rewrites + re-critique
                                    └─► Dataset Builder   versioned benchmark JSON
                                         └─► Answer Verifier   RDKit + PubChem ground truth
                                              └─► Model Evaluator   score any LLM
```

### Agent roles

| Agent | File | Role |
|-------|------|------|
| A1 Segment Selector | `segment_selector.py` | Scores paper segments via novelty prior × empirical reward (UCB). Focuses question generation on results/conclusion segments with high chemical entity density. |
| A2 Quality Evaluator | `paper_quality_evaluator.py` | Filters out supplementary materials, datasets, and non-research documents. |
| A3 PubChem Linker | `pubchem_linker.py` | Resolves named compounds to PubChem CIDs and prefetches 13 properties (formula, mass, SMILES, XLogP3, TPSA, etc.). |
| A4 Question Proposer | `question_proposer.py` | Generates T1/T2/T3 candidate questions anchored to novel compounds with verified structures. |
| A5 Tri-Critic | `critics.py` | Three sequential critics (see below). |
| A6 Question Repairer | `question_repairer.py` | Applies structured critic feedback to rewrite and re-submit flagged questions. |

### Question types

| Type | Description | Verification |
|------|-------------|--------------|
| **T1** | PubChem property query — molecular formula, monoisotopic mass, XLogP3, TPSA, H-bond counts | PubChem PUG REST API |
| **T2** | RDKit structural computation on an embedded SMILES — rotatable bonds, aromatic atoms, ring count, stereocenters | RDKit |
| **T3** | Contrastive: two SMILES/CID-identified compounds, asks which has the higher/lower computable property | RDKit on both |

### Tri-Critic design

| Critic | Checks | Approach |
|--------|--------|----------|
| **Critic 1 — Ill-Defined** | Ambiguity, unstated assumptions, self-containment | Dual-solver: Strict (no assumptions) vs Helpful (minimal assumptions). Fails if they disagree. |
| **Critic 2 — Missing Conditions** | Salt/tautomer not specified, property definition unclear, T3 compounds not fully identified | Outputs structured `missing_conditions: [...]` patch list. |
| **Critic 3 — Guessable (Blind Solver)** | Can the question be answered without the paper? | Simulates a solver using only PubChem + RDKit + general knowledge. Kills trivially answerable questions. |

### Segment Novelty Bandit (A1)

Each segment `s` in a paper is scored:

```
UCB(s) = 0.6 × N(s) + 0.4 × R(s) + c × √(ln(n_total + 1) / (n_role + 1))
```

- **N(s)** — novelty prior: rhetorical role weight × chemical entity density × length bonus
- **R(s)** — empirical reward: fraction of questions from this segment type that survived critics
- Exploration bonus encourages trying under-explored segment types

Rhetorical role weights (results=1.0, conclusion=0.95, tables=0.9, discussion=0.85, methods=0.6, abstract=0.5 …). Rewards persist in `data/segment_rewards.json` and improve over time.

## Installation

```bash
git clone https://github.com/SilviAvayan/livechembench
cd livechembench
```

**Core dependencies (agents + verification + evaluation):**
```bash
micromamba create -n chem python=3.10
micromamba activate chem
pip install openai pydantic pyyaml rdkit requests
```

**Segmentation (GPU required, L4 or better recommended):**
```bash
pip install -r requirements-paddleocr-vl.txt
```

**Environment variable:**
```bash
export NVIDIA_API_KEY="your_key_here"   # from build.nvidia.com
```

## Usage

### One-shot scripts

```bash
# Generate questions from all segmented papers → data/benchmark/livechembench_v0.3.0.json
bash run_pipeline.sh 0.3.0

# Evaluate all available NVIDIA-hosted models against the benchmark
bash eval_all_models.sh
```

### Segment PDFs

Place raw PDFs in `data/raw_papers/`, then run:

```bash
# All PDFs
python -m src.main --segment

# Specific papers by filename
python -m src.main --segment --papers paper1.pdf,paper2.pdf

# With batch size and per-PDF timeout (minutes)
python -m src.main --segment --batch-size 4 --timeout 60
```

Output lands in `data/segmented_papers/<paper_id>.json`.

### Run the full benchmark pipeline

```bash
# A1: score + rank segments per paper (novelty bandit)
python -m src.agents.segment_selector --top-k 3

# A2: filter real research papers
python -m src.agents.paper_quality_evaluator

# A3: resolve chemical entities to PubChem CIDs
python -m src.agents.pubchem_linker

# A4: generate T1/T2/T3 candidate questions
python -m src.agents.question_proposer

# A5: tri-critic quality gate
python -m src.agents.critics

# A6: repair and re-critique flagged questions
python -m src.agents.question_repairer

# Build versioned benchmark
python -m src.agents.dataset_builder --version 0.1.0

# Verify ground-truth answers via RDKit + PubChem
python -m src.agents.answer_verifier \
    --benchmark data/benchmark/livechembench_v0.1.0.json

# Drop failed verifications and re-build
python -m src.agents.dataset_builder --version 0.1.1 \
    --verification-report data/verification/livechembench_v0.1.0_verified.json
```

Every agent supports `--paper-id <id>` to process a single paper and `--limit N` to cap the run.

### Evaluate an LLM

```bash
# Evaluate a single model (default benchmark: latest v0.3.0)
python -m src.agents.model_evaluator \
    --model nvidia/openai/gpt-oss-120b

# Run all available models in one shot
bash eval_all_models.sh

# Use OpenAI / other OpenAI-compatible endpoints
python -m src.agents.model_evaluator \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key-env OPENAI_API_KEY

# View results
ls data/eval_results/*.json | xargs -I{} jq '{model:.model, acc:.scores.overall, n:.scores.n_total}' {}
```

Output: `data/eval_results/v<version>_<model_slug>.json` with accuracy broken down by question type (T1/T2/T3), answer type, and source paper.

#### Available evaluation models

The pipeline agent uses `gcp/google/gemini-3.1-flash-lite-preview`; do **not** use that model for evaluation.

| Model | Provider | Notes |
|-------|----------|-------|
| `nvidia/openai/gpt-oss-120b` | NVIDIA | GPT-OSS 120B |
| `nvidia/qwen/qwen3-next-80b-a3b-instruct` | NVIDIA | Qwen3 80B MoE |
| `nvidia/nvidia/Nemotron-3-Nano-30B-A3B` | NVIDIA | Nemotron 30B MoE |
| `gpt-4o`, `gpt-4o-mini` | OpenAI | Needs `OPENAI_API_KEY` |
| Claude (any) | Anthropic | Run manually via claude.ai or Anthropic SDK |

## Project structure

```
livechembench/
├── src/
│   ├── agents/
│   │   ├── models.py                  # all Pydantic v2 models
│   │   ├── segment_selector.py        # A1: novelty-bandit segment scoring
│   │   ├── paper_quality_evaluator.py # A2: filter real research papers
│   │   ├── pubchem_linker.py          # A3: entity → PubChem CID resolution
│   │   ├── question_proposer.py       # A4: generate T1/T2/T3 candidates
│   │   ├── critics.py                 # A5: tri-critic (ill-defined, conditions, novelty)
│   │   ├── question_repairer.py       # A6: rewrite + re-critique
│   │   ├── novelty_selector.py        # legacy novelty filter (superseded by Critic 3)
│   │   ├── dataset_builder.py
│   │   ├── answer_verifier.py
│   │   └── model_evaluator.py
│   └── services/
│       ├── segment_pipeline.py
│       └── segmenter.py
├── prompts/                           # YAML configs for every LLM agent
│   ├── paper_quality_evaluator.yaml
│   ├── question_proposer.yaml
│   ├── critic_ill_defined.yaml        # Critic 1: dual-solver ambiguity check
│   ├── critic_missing_conditions.yaml # Critic 2: structured missing-conditions list
│   ├── critic_novelty.yaml            # Critic 3: blind-solver guessability check
│   ├── question_repairer.yaml
│   └── model_evaluator.yaml
└── data/
    ├── segment_rewards.json           # empirical reward history for bandit
    ├── segment_selections/            # A1 output: ranked segments per paper
    ├── pubchem_links/                 # A3 output: resolved CIDs + properties
    └── benchmark/                    # versioned benchmark JSONs (tracked in git)
```

## Current benchmark status

**v0.3.0** — 7 verified questions from 4 papers (2 PMC, 2 PubMed):

| ID | Type | Compound | Question |
|----|------|----------|---------|
| 2026-05_001 | T2 | kaempferol | Rotatable bonds (SMILES embedded) |
| 2026-05_002 | T3 | arbutin vs kaempferol | Which has more aromatic atoms? |
| 2026-05_003 | T2 | andrographolide | Number of stereocenters |
| 2026-05_004 | T2 | Istaroxime | Rotatable bonds |
| 2026-05_005 | T2 | Istaroxime | Aromatic atoms |
| 2026-05_006 | T2 | Bis-GMA | Rotatable bonds |
| 2026-05_007 | T3 | TEGDMA vs UDMA | Which has more rotatable bonds? |

All answers are ground-truthed by RDKit. The dataset grows automatically as more papers are segmented and pass the full pipeline.

---

## Proposer-bias experiment (recommended follow-up after the presentation)

After the initial presentation we ran a controlled experiment to test whether benchmark scores are
inflated for whichever model *generated* the questions — a form of **proposer-model bias** that
could make a leaderboard misleading.

### What we did

We regenerated questions for the same 4 papers that contributed to `v0.3.0`, but swapped the
proposer from **Gemini Flash-Lite** to **`nvidia/openai/gpt-oss-120b`**, producing benchmark
**`v0.6.0`**. All downstream agents (critics, repairer, novelty selector, answer verifier) were
kept on Gemini Flash-Lite to isolate the proposer as the only changed variable. We then evaluated
the same 4 models on both benchmarks.

### Results

| Model | v0.3.0 (Gemini-proposed, n=7) | v0.6.0 (GPT-proposed, n=2) | Δ |
|---|---|---|---|
| **Gemini Flash-Lite** | **85.7%** | 0.0% | −85.7% |
| GPT-oss-120b | 42.9% | 0.0% | −42.9% |
| Nemotron-30B | 28.6% | 0.0% | −28.6% |
| **Qwen3-80b** | 14.3% | **50.0%** | +35.7% |

The leaderboard **inverts** between the two benchmarks. Gemini Flash-Lite (the v0.3.0 proposer)
scores 85.7% on its own questions but drops to 0% on GPT-generated questions. Qwen3-80b — weakest
on the Gemini benchmark — becomes the strongest on the GPT benchmark. This is the textbook
signature of proposer-model bias.

### Interpretation

The results suggest that a model scoring highly on LiveChemBench may partly reflect alignment
between its reasoning style and the proposer's question-framing style, rather than pure chemistry
knowledge. Mitigations for a production benchmark include:
- Rotating the proposer across multiple model families each month.
- Using a neutral third-party model as proposer (distinct from all evaluated models).
- Averaging scores across proposer variants.

### Caveat

`v0.6.0` contains only 2 questions because the `QuestionType` enum accepted only T1–T3 at run
time — GPT's T4/T5 proposals were silently rejected. The enum is now extended to T1–T6 in the
`experiment/gpt-proposer` branch. A full re-run with the fix would produce a statistically
meaningful sample (target ≥ 30 questions). This is the primary recommended follow-up.

### Reproducing the experiment

```bash
git checkout experiment/gpt-proposer
export NVIDIA_API_KEY=<your key>
bash run_experiment_gpt.sh                    # ~3 minutes for 4 papers
python experiments/compare_proposer_bias.py   # prints the leaderboard table
```

Full experiment design and discussion: [`experiments/README.md`](experiments/README.md).

---

## Benchmark format

`data/benchmark/livechembench_v<X.Y.Z>.json`:

```json
{
  "name": "LiveChemBench",
  "version": "0.3.0",
  "created_at": "2026-05-04T14:56:31+00:00",
  "stats": {"total": 7, "by_type": {"T2": 5, "T3": 2}},
  "questions": [
    {
      "id": "2026-05_001",
      "paper_id": "source:pmc:12987923",
      "segment_id": "abstract",
      "cid": 3467,
      "question": "How many rotatable bonds are present in kaempferol (SMILES: ...)...",
      "answer": "1",
      "answer_type": "int",
      "tolerance": null,
      "verifier": {
        "type": "rdkit",
        "recipe": { "function": "Chem.rdMolDescriptors", "input": "smiles", "description": "..." }
      },
      "filters": { "ill_defined": false, "missing_conditions": [], "guessable": false },
      "provenance": { "month": "2026-05", "paper_source": "pmc", "conversion_tool": "paddle_vl", "pubchem_query_log_hash": "..." },
      "question_type": "T2",
      "chemical_entities": ["kaempferol"]
    }
  ]
}
```

## LLM backend

All agents use `gcp/google/gemini-3.1-flash-lite-preview` by default via the NVIDIA Inference API (`https://inference-api.nvidia.com/v1`). Override any agent's model by editing its YAML in `prompts/` or passing `--model` to the evaluator.

## License

MIT
