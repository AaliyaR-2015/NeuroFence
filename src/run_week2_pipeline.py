"""
run_week2_pipeline.py
-----------------------
Integration entry point for Week 2. Depends on all four modules, so -- like
main.py and test_harness.py in Week 1 -- it lives on `main`, assembled once
every member branch is merged in.

Produces the two artifacts the rest of the project builds on:
    outputs/baseline_profile.json   (Member 3's NeuronBaselineProfile,
                                      built from benign-only prompts)
    outputs/fuzz_sweep_log.json     (Member 2's mixed-category sweep,
                                      what the Week 2 UI heatmap reads)

Usage
-----
    python -m src.run_week2_pipeline --model sshleifer/tiny-gpt2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.activation_tracker import ActivationTracker
from src.fuzz_runner import build_baseline_profile, run_full_fuzz_sweep, save_sweep_log
from src.model_sandbox import UnsafeModelError, load_model_safely


def run(
    model_name: str,
    n_benign_baseline: int,
    n_benign_sweep: int,
    n_edge_case: int,
    n_trigger_candidate: int,
    seed: int,
    output_dir: Path,
) -> None:
    print(f"[1/5] Loading '{model_name}' into the sandbox...")
    try:
        model, tokenizer, metadata = load_model_safely(model_name)
    except UnsafeModelError as exc:
        print(f"REFUSED TO LOAD: {exc}")
        raise SystemExit(1)
    print(f"      architecture={metadata.architecture} "
          f"params={metadata.num_parameters:,} layers={metadata.num_layers} "
          f"weights_sha256={metadata.weights_sha256[:16]}...")

    print("[2/5] Attaching activation hooks (per-neuron tracking enabled)...")
    tracker = ActivationTracker(model, track_neurons=True)
    tracker.attach()
    print(f"      {len(tracker._handles)} transformer block(s) instrumented")

    print(f"[3/5] Building baseline profile from {n_benign_baseline} benign prompts...")
    baseline = build_baseline_profile(model, tokenizer, tracker, n_benign=n_benign_baseline, seed=seed)
    baseline_path = output_dir / "baseline_profile.json"
    baseline.save(baseline_path)
    n_layers = len(baseline.finalize())
    print(f"      baseline covers {n_layers} layer(s) -> {baseline_path}")

    print(f"[4/5] Running full fuzz sweep "
          f"({n_benign_sweep} benign / {n_edge_case} edge-case / {n_trigger_candidate} trigger-candidate)...")
    sweep_results = run_full_fuzz_sweep(
        model, tokenizer, tracker,
        n_benign=n_benign_sweep, n_edge_case=n_edge_case, n_trigger_candidate=n_trigger_candidate,
        seed=seed + 1,  # different seed from baseline -- sweep prompts shouldn't be identical to baseline prompts
    )
    sweep_path = output_dir / "fuzz_sweep_log.json"
    save_sweep_log(sweep_results, sweep_path)
    print(f"      {len(sweep_results)} prompt(s) run -> {sweep_path}")

    tracker.detach()
    print("[5/5] Done. Load outputs/fuzz_sweep_log.json in the desktop app's "
          "'Load Fuzz Sweep Log...' button to see the heatmap.")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroFence Week 2 pipeline")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--n-benign-baseline", type=int, default=200)
    parser.add_argument("--n-benign-sweep", type=int, default=50)
    parser.add_argument("--n-edge-case", type=int, default=25)
    parser.add_argument("--n-trigger-candidate", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run(
        model_name=args.model,
        n_benign_baseline=args.n_benign_baseline,
        n_benign_sweep=args.n_benign_sweep,
        n_edge_case=args.n_edge_case,
        n_trigger_candidate=args.n_trigger_candidate,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
