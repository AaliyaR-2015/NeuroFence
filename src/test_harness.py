"""
test_harness.py
----------------
CLI entry point that proves the Week 1 pipeline works end-to-end:

    load a model safely -> attach activation hooks -> run prompts through it
    -> dump per-layer activation stats to a JSON log.

This is deliberately a thin script, not a library: its only job is to be a
runnable, demoable proof that model_sandbox.py and activation_tracker.py
work together. The real fuzzing/detection logic (Weeks 2-3) will build on
top of this, not inside it.

Usage
-----
    python -m src.test_harness --model sshleifer/tiny-gpt2
    python -m src.test_harness --model gpt2 --prompts "Hello world" "The capital of France is"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.activation_tracker import ActivationTracker
from src.model_sandbox import UnsafeModelError, load_model_safely

DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a distant galaxy, an AI system woke up and",
    "def add(a, b):",
]


def run(model_name: str, prompts: list[str], output_path: Path) -> None:
    print(f"[1/4] Loading '{model_name}' into the sandbox...")
    try:
        model, tokenizer, metadata = load_model_safely(model_name)
    except UnsafeModelError as exc:
        print(f"REFUSED TO LOAD: {exc}")
        raise SystemExit(1)

    print(f"      architecture={metadata.architecture} "
          f"params={metadata.num_parameters:,} layers={metadata.num_layers}")

    print("[2/4] Attaching activation hooks...")
    tracker = ActivationTracker(model)
    tracker.attach()
    n_hooks = len(tracker._handles)
    print(f"      {n_hooks} transformer block(s) instrumented")

    print(f"[3/4] Running {len(prompts)} prompt(s) through the model...")
    all_runs = []
    for prompt in prompts:
        tracker.clear()
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
        all_runs.append(
            {
                "prompt": prompt,
                "layer_stats": [s.as_dict() for s in tracker.last_run_stats()],
            }
        )

    tracker.detach()

    print(f"[4/4] Writing activation log to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"model_metadata": metadata.as_dict(), "runs": all_runs},
            indent=2,
        )
    )
    print("Done. Week 1 pipeline (sandbox -> hooks -> logged activations) verified.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroFence Week 1 test harness")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2",
                         help="HF model id or local path (must have safetensors weights)")
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS,
                         help="Prompts to run through the model")
    parser.add_argument("--output", default="outputs/activation_log.json",
                         help="Where to write the JSON activation log")
    args = parser.parse_args()

    run(args.model, args.prompts, Path(args.output))


if __name__ == "__main__":
    main()
