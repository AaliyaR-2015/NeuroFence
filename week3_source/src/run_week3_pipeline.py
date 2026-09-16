"""
run_week3_pipeline.py — Week 3 integration

Ties together (all four modules, same pattern as run_week2_pipeline.py):
  Member 1  model_sandbox        -> load a model safely
  Member 2  adversarial_fuzzer   -> generate_batch() of benign/edge_case/trigger_candidate prompts
  Member 3  activation_tracker + baseline_profile + anomaly_detector (this week's new piece)
  Member 4  (separate: ui/anomaly_report_widget.py renders outputs/anomaly_report.json)

Usage:
    # 1. Build (or reuse) a Week 2 baseline profile first:
    python -m src.run_week2_pipeline --model sshleifer/tiny-gpt2

    # 2. Create a known-positive test fixture (optional but recommended --
    #    this is what proves the scanner actually catches something):
    python -c "from src.backdoor_injector import create_test_backdoored_model; \
        create_test_backdoored_model('sshleifer/tiny-gpt2', 'outputs/poisoned_test_model')"

    # 3. Scan the poisoned model against the clean baseline:
    python -m src.run_week3_pipeline --baseline outputs/baseline_profile.json \
        --model outputs/poisoned_test_model

    # ...or scan the clean model itself, as a negative control (should find
    # ~nothing, since it IS what the baseline was built from):
    python -m src.run_week3_pipeline --baseline outputs/baseline_profile.json \
        --model sshleifer/tiny-gpt2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.adversarial_fuzzer import generate_batch
from src.activation_tracker import ActivationTracker
from src.anomaly_detector import (
    detect_anomalous_neurons_all_layers,
    load_baseline,
    save_anomaly_report,
)

# --- ADAPT ME -----------------------------------------------------------
# model_sandbox.py's exact loader function name/signature isn't reproduced
# here (this file was written without direct access to your source) —
# copy the 2-3 lines you already use in run_week2_pipeline.py to load a
# model + tokenizer safely, and drop them into `load_model()` below.
def load_model(model_path_or_id: str):
    from src.model_sandbox import load_model_safely  # <- match your real function name
    return load_model_safely(model_path_or_id)
# --------------------------------------------------------------------------


def collect_per_neuron_means_by_category(model, tokenizer, n_per_category: int = 64, seed: int = 0):
    """Runs each fuzz category through the model with track_neurons=True and
    returns { layer_name: { category: [[neuron_means...], ...] } }."""
    categories = ["benign", "edge_case", "trigger_candidate"]
    prompts_by_category = generate_batch(n_per_category=n_per_category, seed=seed)

    per_layer_category_means: dict = defaultdict(lambda: defaultdict(list))

    with ActivationTracker(model, track_neurons=True) as tracker:
        for category in categories:
            prompts = prompts_by_category[category]
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                model(**inputs)
                stats = tracker.last_stats()  # { layer_name: LayerActivationStats }
                for layer_name, layer_stats in stats.items():
                    if layer_stats.neuron_means is None:
                        continue
                    per_layer_category_means[layer_name][category].append(
                        np.asarray(layer_stats.neuron_means).tolist()
                    )

    return {layer: dict(cats) for layer, cats in per_layer_category_means.items()}


def main():
    parser = argparse.ArgumentParser(description="NeuroFence Week 3: anomaly scan against baseline")
    parser.add_argument("--baseline", required=True, help="Path to outputs/baseline_profile.json")
    parser.add_argument("--model", required=True, help="HF model id or local safetensors directory")
    parser.add_argument("--n-per-category", type=int, default=64)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--out", default="outputs/anomaly_report.json")
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    model, tokenizer = load_model(args.model)

    per_layer_category_means = collect_per_neuron_means_by_category(
        model, tokenizer, n_per_category=args.n_per_category
    )

    findings = detect_anomalous_neurons_all_layers(
        baseline, per_layer_category_means, z_threshold=args.z_threshold
    )

    save_anomaly_report(
        findings,
        args.out,
        meta={"model": args.model, "baseline": args.baseline, "z_threshold": args.z_threshold},
    )

    selective = [f for f in findings if f.is_selective_trigger]
    print(f"Scanned {args.model}")
    print(f"  {len(findings)} anomalous neurons total")
    print(f"  {len(selective)} look like SELECTIVE TRIGGER neurons (dormant on benign, spike on trigger prompts)")
    if selective:
        top = selective[0]
        print(f"  Most suspicious: {top.layer} neuron #{top.neuron_idx} "
              f"(benign z={top.benign_z:.2f}, trigger z={top.trigger_z:.2f})")
    print(f"Full report written to {args.out}")


if __name__ == "__main__":
    main()
