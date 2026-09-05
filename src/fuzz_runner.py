"""
fuzz_runner.py
---------------
Week 2 scope, per the project plan: "Build the adversarial text generator.
Feed thousands of prompts into the model and record the baseline
activation patterns of the neurons."

This module is the integration point between:
    - adversarial_fuzzer.py  (this branch)   -> generates prompts
    - model_sandbox.py       (Member 1)      -> loads the model safely,
                                                batch-tokenizes prompts
    - activation_tracker.py  (Member 3)      -> captures per-neuron
                                                activations via hooks
    - baseline_profile.py    (Member 3)      -> accumulates the running
                                                per-neuron baseline

It deliberately does NOT implement anomaly detection itself (Week 3) --
its job is only to produce two artifacts:
    1. A NeuronBaselineProfile built from *benign* prompts only (since a
       baseline must represent "normal" behavior).
    2. A full sweep log covering benign + edge_case + trigger_candidate
       prompts, tagged by category, with per-layer scalar stats -- this is
       what the Week 2 UI heatmap (Member 4) visualizes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.activation_tracker import ActivationTracker
from src.adversarial_fuzzer import FuzzPrompt, generate_batch
from src.baseline_profile import NeuronBaselineProfile
from src.model_sandbox import tokenize_batch

DEFAULT_BATCH_SIZE = 16


def build_baseline_profile(
    model: torch.nn.Module,
    tokenizer: Any,
    tracker: ActivationTracker,
    n_benign: int = 200,
    seed: int | None = 42,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NeuronBaselineProfile:
    """
    Feed `n_benign` benign prompts through the model in batches, and
    accumulate a NeuronBaselineProfile from their per-neuron activations.

    `tracker` must have been constructed with `track_neurons=True` (this
    function does not create the tracker itself, so the same tracker
    instance -- and therefore the same set of hooked layers -- can be
    reused for both baseline-building and the full sweep below).
    """
    prompts = generate_batch(n_benign=n_benign, n_edge_case=0, n_trigger_candidate=0, seed=seed)
    profile = NeuronBaselineProfile()

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        _run_batch_through_model(model, tokenizer, tracker, batch, profile=profile)

    return profile


def run_full_fuzz_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    tracker: ActivationTracker,
    n_benign: int = 50,
    n_edge_case: int = 25,
    n_trigger_candidate: int = 25,
    seed: int | None = 123,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """
    Run a mixed batch (benign + edge_case + trigger_candidate) through the
    model and return one log entry per prompt:
        {"prompt": str, "category": str, "layer_stats": [...]}

    This is the raw material the Week 2 UI heatmap aggregates (mean
    activation per layer per category) -- it intentionally keeps every
    prompt's result rather than pre-aggregating, so Week 3's anomaly
    scoring can also reuse it without re-running the model.
    """
    prompts = generate_batch(
        n_benign=n_benign,
        n_edge_case=n_edge_case,
        n_trigger_candidate=n_trigger_candidate,
        seed=seed,
    )

    results: list[dict] = []
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        results.extend(_run_batch_through_model(model, tokenizer, tracker, batch, profile=None))

    return results


def _run_batch_through_model(
    model: torch.nn.Module,
    tokenizer: Any,
    tracker: ActivationTracker,
    batch: list[FuzzPrompt],
    profile: NeuronBaselineProfile | None,
) -> list[dict]:
    """
    Shared inner loop: tokenize a batch, run it through the model one
    prompt at a time (hooks fire per forward call, and HF attention masks
    make truly batched per-prompt attribution fiddly -- Week 1's
    test_harness already established the one-prompt-per-forward-call
    pattern, so this keeps that proven approach rather than introducing
    batched-then-unbatched bookkeeping risk this early).

    If `profile` is given, every prompt's neuron means are folded into it
    (baseline-building mode). Otherwise, per-prompt results are returned
    for logging (full-sweep mode).
    """
    # tokenize_batch still validates/pads consistently even though we then
    # iterate one row at a time -- keeps tokenization behavior identical
    # between baseline-building and full-sweep runs.
    tokenize_batch(tokenizer, [p.text for p in batch])

    entries = []
    for prompt in batch:
        tracker.clear()
        inputs = tokenizer(prompt.text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(**inputs)
        stats = tracker.last_run_stats()

        if profile is not None:
            for stat in stats:
                if stat.neuron_means is not None:
                    profile.update(stat.layer_name, stat.neuron_means)
        else:
            entries.append(
                {
                    "prompt": prompt.text,
                    "category": prompt.category,
                    "layer_stats": [
                        {k: v for k, v in s.as_dict().items() if k != "neuron_means"}
                        for s in stats
                    ],
                }
            )

    return entries


def save_sweep_log(entries: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": entries}, indent=2))
