# LiveChemBench

[![CI](https://github.com/SilviAvayan/livechembench/actions/workflows/ci.yml/badge.svg)](https://github.com/SilviAvayan/livechembench/actions/workflows/ci.yml)

A dynamic, contamination-resistant chemistry reasoning benchmark built automatically from recent peer-reviewed literature. The pipeline ingests raw chemistry PDFs, generates structured benchmark questions, enforces quality through a multi-agent critic loop, verifies answers deterministically using RDKit and PubChem, and evaluates LLMs against the resulting dataset — all without human annotation.

## Motivation

Static chemistry benchmarks become unreliable over time as LLMs train on publicly available test questions. LiveChemBench addresses this by generating fresh questions from newly published papers on demand, making contamination structurally difficult.

## Pipeline

```
PDFs
 └─► Segmentation (PaddleOCR-VL)
      └─► Quality Evaluator         filters for real research papers
           └─► Question Proposer    generates T1 / T2 / T3 questions
                └─► Critics 1 & 2  ill-defined + missing-conditions checks
                     └─► Repairer  LLM rewrites + re-critique loop
                          └─► Novelty Selector   computation-required filter
                               └─► Dataset Builder  versioned benchmark JSON
                                    └─► Answer Verifier  RDKit + PubChem ground truth
                                         └─► Model Evaluator  score any LLM
```

### Question types

| Type | Description | Verification |
|------|-------------|--------------|
| **T1** | PubChem property lookup — molecular formula, monoisotopic mass, XLogP3, TPSA | PubChem PUG REST API |
| **T2** | RDKit structural computation on an embedded SMILES string — rotatable bonds, aromatic atoms, ring count | RDKit |
| **T3** | Comparative: two SMILES-identified compounds, asks which has higher/lower computable property | RDKit on both |

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

### Segment PDFs

Place raw PDFs in `data/raw_papers/`, then run:

```bash
# All PDFs in data/raw_papers/
python -m src.main

# Specific PDFs by filename
python scripts/segment_explicit_pdfs.py paper1.pdf paper2.pdf

# With batch size and per-PDF timeout (seconds)
python scripts/segment_explicit_pdfs.py --batch-size 4 --timeout 300 paper1.pdf paper2.pdf
```

Output lands in `data/segmented_papers/<paper_id>.json`.

### Run the benchmark pipeline

```bash
python -m src.agents.paper_quality_evaluator   # filter: real research papers only
python -m src.agents.question_proposer         # generate T1/T2/T3 candidates
python -m src.agents.critics                   # ill-defined + missing-conditions checks
python -m src.agents.question_repairer         # repair flagged questions + re-critique
python -m src.agents.novelty_selector          # drop memory-answerable questions
python -m src.agents.dataset_builder --version 0.1.0
python -m src.agents.answer_verifier --benchmark data/benchmark/livechembench_v0.1.0.json
python -m src.agents.dataset_builder --version 0.1.1 \
    --verification-report data/verification/livechembench_v0.1.0_verified.json
```

Every agent supports `--paper-id <id>` to process a single paper and `--limit N` to cap the run.

### Evaluate an LLM

```bash
python -m src.agents.model_evaluator \
    --benchmark data/benchmark/livechembench_v0.1.1.json

# Override the model
python -m src.agents.model_evaluator \
    --benchmark data/benchmark/livechembench_v0.1.1.json \
    --model gcp/google/gemini-2.0-flash
```

Output: `data/eval_results/v<version>_<model>.json` with accuracy broken down by question type (T1/T2/T3), answer type, and source paper.

## Project structure

```
livechembench/
├── src/
│   ├── agents/
│   │   ├── models.py                  # all Pydantic v2 models
│   │   ├── paper_quality_evaluator.py
│   │   ├── question_proposer.py
│   │   ├── critics.py
│   │   ├── question_repairer.py
│   │   ├── novelty_selector.py
│   │   ├── dataset_builder.py
│   │   ├── answer_verifier.py
│   │   └── model_evaluator.py
│   └── services/
│       ├── segment_pipeline.py
│       └── segmenter.py
├── prompts/                           # YAML configs for every LLM agent
│   ├── paper_quality_evaluator.yaml
│   ├── question_proposer.yaml
│   ├── critic_ill_defined.yaml
│   ├── critic_missing_conditions.yaml
│   ├── critic_novelty.yaml
│   ├── question_repairer.yaml
│   └── model_evaluator.yaml
├── scripts/
│   └── segment_explicit_pdfs.py      # segment a named list of PDFs
└── data/
    └── benchmark/                     # versioned benchmark JSONs (tracked in git)
```

## Benchmark format

`data/benchmark/livechembench_v<X.Y.Z>.json`:

```json
{
  "name": "LiveChemBench",
  "version": "0.1.1",
  "created_at": "2026-05-03T21:36:20+00:00",
  "stats": {"total": 3, "by_type": {"T1": 1, "T2": 1, "T3": 1}},
  "questions": [
    {
      "id": "lcb_0001",
      "paper_id": "pubmed_12478036",
      "question_text": "...",
      "answer": "...",
      "answer_type": "float",
      "answer_units": "Da",
      "tolerance": 0.01,
      "question_type": "T1",
      "chemical_entities": ["Nec-1s"],
      "verification_recipe": "...",
      "source_segment": "abstract"
    }
  ]
}
```

## LLM backend

All agents use `gcp/google/gemini-3.1-flash-lite-preview` by default via the NVIDIA Inference API (`https://inference-api.nvidia.com/v1`). Override any agent's model by editing its YAML in `prompts/` or passing `--model` to the evaluator.

## License

MIT
