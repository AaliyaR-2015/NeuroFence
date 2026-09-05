"""
baseline_profile.py
--------------------
Accumulates a running per-neuron mean/variance profile, per layer, across
however many prompts are fed through the model -- without ever holding all
of those prompts' activations in memory at once.

Why Welford's algorithm: Week 2 needs to run "thousands of prompts" (per
the project plan) through the model to build a baseline of "normal"
per-neuron behavior. Naively, that means storing a (num_prompts x
hidden_size) array per layer and calling numpy.mean/std on it afterwards --
fine for a demo, wasteful (and eventually a memory problem) for a real
security tool meant to run large fuzzing sweeps. Welford's algorithm
computes an exact running mean and variance in a single pass, using only
O(hidden_size) memory per layer regardless of how many prompts are fed in.

This profile is the "what does normal look like" baseline that Week 3's
anomaly/backdoor detection will compare new activations against (e.g. via
a z-score: how many standard deviations is this neuron's activation from
its baseline mean, for a given input).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NeuronBaselineProfile:
    """
    Per-layer, per-neuron running mean/variance, updated one sample
    (one forward pass's neuron-mean vector) at a time via Welford's
    online algorithm.

    Usage
    -----
        profile = NeuronBaselineProfile()
        for prompt in many_benign_prompts:
            ...  # run prompt through model + tracker
            for stat in tracker.last_run_stats():
                profile.update(stat.layer_name, stat.neuron_means)
        summary = profile.finalize()
    """

    def __init__(self) -> None:
        # layer_name -> {"n": int, "mean": np.ndarray, "m2": np.ndarray}
        self._state: dict[str, dict[str, Any]] = {}

    def update(self, layer_name: str, neuron_means: list[float]) -> None:
        """Incorporate one new sample (one forward pass) for a layer."""
        x = np.asarray(neuron_means, dtype=np.float64)

        if layer_name not in self._state:
            self._state[layer_name] = {
                "n": 0,
                "mean": np.zeros_like(x),
                "m2": np.zeros_like(x),
            }

        state = self._state[layer_name]
        if state["mean"].shape != x.shape:
            raise ValueError(
                f"Shape mismatch for layer '{layer_name}': profile expects "
                f"{state['mean'].shape}, got {x.shape}. Are you mixing "
                f"models with different hidden sizes in one profile?"
            )

        # Welford's online update.
        state["n"] += 1
        delta = x - state["mean"]
        state["mean"] += delta / state["n"]
        delta2 = x - state["mean"]
        state["m2"] += delta * delta2

    def finalize(self) -> dict[str, dict[str, Any]]:
        """
        Return the final per-layer summary:
            {layer_name: {"mean": [...], "std": [...], "n": int}}

        Layers seen only once have std=0 for every neuron (variance is
        undefined with a single sample; treating it as zero is the
        conservative choice for Week 3's z-score detector, since it will
        flag *any* deviation as maximally anomalous rather than divide by
        zero).
        """
        summary = {}
        for layer_name, state in self._state.items():
            n = state["n"]
            variance = state["m2"] / n if n > 0 else np.zeros_like(state["m2"])
            std = np.sqrt(variance)
            summary[layer_name] = {
                "mean": state["mean"].tolist(),
                "std": std.tolist(),
                "n": n,
            }
        return summary

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.finalize(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NeuronBaselineProfile":
        """
        Load a previously-saved profile back into a NeuronBaselineProfile
        so more samples can, in principle, be added later -- though most
        callers (Week 3 onward) will just read `.finalize()`-shaped data
        directly rather than resuming accumulation.
        """
        data = json.loads(Path(path).read_text())
        profile = cls()
        for layer_name, layer_data in data.items():
            n = layer_data["n"]
            mean = np.asarray(layer_data["mean"], dtype=np.float64)
            std = np.asarray(layer_data["std"], dtype=np.float64)
            profile._state[layer_name] = {
                "n": n,
                "mean": mean,
                "m2": (std**2) * n,
            }
        return profile
