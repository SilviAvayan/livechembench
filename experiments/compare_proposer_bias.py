"""
Compare model leaderboards between the Gemini-proposed (v0.3.0) and
GPT-proposed (v0.6.0) benchmarks to detect proposer-model bias.

Usage:
    python experiments/compare_proposer_bias.py

Expected outputs:
    - Leaderboard table printed to stdout
    - experiments/proposer_bias_results.json   (machine-readable)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MODEL_DISPLAY = {
    "gcp/google/gemini-3.1-flash-lite-preview": "Gemini Flash-Lite",
    "nvidia/openai/gpt-oss-120b":               "GPT-oss-120b",
    "nvidia/nvidia/Nemotron-3-Nano-30B-A3B":    "Nemotron-30B",
    "nvidia/qwen/qwen3-next-80b-a3b-instruct":  "Qwen3-80b",
}


def load_eval(path: Path) -> tuple[str, float, int]:
    """Return (model, accuracy, n_questions)."""
    d = json.loads(path.read_text())
    model = d.get("model", "unknown")
    results = d.get("results", [])
    if not results:
        return model, 0.0, 0
    correct = sum(1 for r in results if r.get("correct"))
    return model, correct / len(results), len(results)


def collect(version: str) -> dict[str, tuple[float, int]]:
    """Return {model: (accuracy, n)} for all eval files of a given version."""
    out = {}
    for f in sorted((REPO / "data" / "eval_results").glob(f"v{version}_*.json")):
        model, acc, n = load_eval(f)
        out[model] = (acc, n)
    return out


def main() -> None:
    gemini_evals = collect("0.3.0")
    gpt_evals    = collect("0.6.0")

    all_models = sorted(set(gemini_evals) | set(gpt_evals))

    if not gemini_evals:
        print("No v0.3.0 eval results found — run model_evaluator on v0.3.0 first.")
        sys.exit(1)
    if not gpt_evals:
        print("No v0.6.0 eval results found — run run_experiment_gpt.sh first.")
        sys.exit(1)

    # ── Print table ──────────────────────────────────────────────────────────
    header = f"{'Model':<24}  {'v0.3.0 (Gemini proposer)':>24}  {'v0.6.0 (GPT proposer)':>22}  {'Δ':>8}"
    print()
    print("Proposer-Bias Experiment — Leaderboard Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = []
    for m in all_models:
        label = MODEL_DISPLAY.get(m, m)
        g_acc, g_n = gemini_evals.get(m, (None, 0))
        p_acc, p_n = gpt_evals.get(m, (None, 0))

        g_str = f"{g_acc:.1%} (n={g_n})" if g_acc is not None else "—"
        p_str = f"{p_acc:.1%} (n={p_n})" if p_acc is not None else "—"

        if g_acc is not None and p_acc is not None:
            delta = p_acc - g_acc
            d_str = f"{delta:+.1%}"
        else:
            delta = None
            d_str = "—"

        print(f"{label:<24}  {g_str:>24}  {p_str:>22}  {d_str:>8}")
        rows.append({"model": m, "v0.3.0_acc": g_acc, "v0.6.0_acc": p_acc, "delta": delta})

    print("=" * len(header))
    print()
    print("Interpretation:")
    print("  If the proposer model scores notably HIGHER on its own benchmark,")
    print("  that indicates proposer-model bias in the generated questions.")
    print("  A consistent ranking across both benchmarks suggests bias is minimal.")
    print()

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = REPO / "experiments" / "proposer_bias_results.json"
    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
