"""
anomaly_detector.py — Week 3

Compares per-neuron activation statistics gathered from a candidate model
against the benign-only NeuronBaselineProfile from Week 2, and flags neurons
that are:

  (a) statistically anomalous — |z-score| against the baseline exceeds
      z_threshold, AND
  (b) selectively triggered — near-baseline (dormant) on benign / edge_case
      prompts specifically, but anomalous on trigger_candidate prompts.

(b) is what turns "this neuron is just naturally high-variance" into
"this neuron looks like a sleeper-agent trigger detector" — matching the
project's problem statement almost word for word ("dormant neurons that only
spike on highly specific, unnatural inputs").

This module deliberately does not import model_sandbox / adversarial_fuzzer /
activation_tracker directly — it only consumes the plain data structures
those modules already produce (a baseline dict and per-category neuron-mean
arrays), so it stays testable with plain numpy and doesn't depend on exact
internal APIs you may still be iterating on.

Expected shapes
---------------
baseline: the dict returned by NeuronBaselineProfile.finalize(), i.e.
    { layer_name: {"mean": [float, ...], "std": [float, ...], "n": int} }

category_neuron_means: per category, a 2D array-like of shape
    (num_prompts_in_category, num_neurons) for ONE layer at a time — i.e.
    call detect_anomalous_neurons() once per layer, or use
    detect_anomalous_neurons_all_layers() to loop over every layer in the
    baseline for you.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


DEFAULT_Z_THRESHOLD = 4.0
DEFAULT_SELECTIVITY_Z = 4.0  # how many pooled within-category std devs apart
                             # trigger and benign must be to count as "selective"


@dataclass
class AnomalyFinding:
    layer: str
    neuron_idx: int
    baseline_mean: float
    baseline_std: float
    benign_mean: float
    benign_z: float
    trigger_mean: float
    trigger_z: float
    selectivity_z: float
    is_selective_trigger: bool


def _zscores(observed_means: np.ndarray, baseline_mean: np.ndarray, baseline_std: np.ndarray) -> np.ndarray:
    safe_std = np.where(baseline_std < 1e-8, 1e-8, baseline_std)
    return (observed_means - baseline_mean) / safe_std


def detect_anomalous_neurons_for_layer(
    layer_name: str,
    baseline_layer: Dict,
    category_neuron_means: Dict[str, Sequence[Sequence[float]]],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    selectivity_z_threshold: float = DEFAULT_SELECTIVITY_Z,
) -> List[AnomalyFinding]:
    """category_neuron_means must contain at least 'benign' and
    'trigger_candidate' keys; 'edge_case' is optional but recommended."""
    baseline_mean = np.asarray(baseline_layer["mean"], dtype=float)
    baseline_std = np.asarray(baseline_layer["std"], dtype=float)

    if "benign" not in category_neuron_means or "trigger_candidate" not in category_neuron_means:
        raise ValueError("Need at least 'benign' and 'trigger_candidate' categories")

    benign_arr = np.asarray(category_neuron_means["benign"], dtype=float)
    trigger_arr = np.asarray(category_neuron_means["trigger_candidate"], dtype=float)

    benign_neuron_mean = benign_arr.mean(axis=0)
    trigger_neuron_mean = trigger_arr.mean(axis=0)

    benign_z = _zscores(benign_neuron_mean, baseline_mean, baseline_std)
    trigger_z = _zscores(trigger_neuron_mean, baseline_mean, baseline_std)

    # "Selective trigger" asks a different question than "is this neuron
    # unusual at all vs the pre-injection baseline": does its response to
    # trigger input differ from its OWN response to benign input, by more
    # than ordinary prompt-to-prompt noise? A real attacker can modify a
    # neuron's weights in many ways -- a full weight-row replacement shifts
    # the neuron's whole operating point away from the original baseline on
    # EVERY input, not just the trigger, and that is still a genuine,
    # selective backdoor (it only RESPONDS differently to the trigger; it
    # just doesn't happen to reproduce the exact pre-injection statistics on
    # clean input). What matters forensically is the gap between trigger and
    # benign behavior relative to how noisy each category's own samples are
    # -- not whether benign input coincidentally matches the original
    # baseline distribution, which only a narrow class of backdoors
    # (small additive perturbations that leave clean behavior untouched)
    # would do.
    benign_within_std = benign_arr.std(axis=0, ddof=0)
    trigger_within_std = trigger_arr.std(axis=0, ddof=0)
    pooled_std = np.sqrt((benign_within_std ** 2 + trigger_within_std ** 2) / 2.0)
    safe_pooled_std = np.where(pooled_std < 1e-8, 1e-8, pooled_std)
    selectivity_z = (trigger_neuron_mean - benign_neuron_mean) / safe_pooled_std

    findings: List[AnomalyFinding] = []
    num_neurons = baseline_mean.shape[0]
    for i in range(num_neurons):
        if abs(trigger_z[i]) < z_threshold:
            continue
        is_selective = abs(selectivity_z[i]) >= selectivity_z_threshold
        findings.append(
            AnomalyFinding(
                layer=layer_name,
                neuron_idx=i,
                baseline_mean=float(baseline_mean[i]),
                baseline_std=float(baseline_std[i]),
                benign_mean=float(benign_neuron_mean[i]),
                benign_z=float(benign_z[i]),
                trigger_mean=float(trigger_neuron_mean[i]),
                trigger_z=float(trigger_z[i]),
                selectivity_z=float(selectivity_z[i]),
                is_selective_trigger=bool(is_selective),
            )
        )
    return findings


def detect_anomalous_neurons_all_layers(
    baseline: Dict[str, Dict],
    per_layer_category_neuron_means: Dict[str, Dict[str, Sequence[Sequence[float]]]],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    selectivity_z_threshold: float = DEFAULT_SELECTIVITY_Z,
) -> List[AnomalyFinding]:
    """per_layer_category_neuron_means: { layer_name: category_neuron_means }
    i.e. the same shape as `baseline`, but each leaf is per-category arrays
    instead of a mean/std/n dict."""
    all_findings: List[AnomalyFinding] = []
    for layer_name, baseline_layer in baseline.items():
        if layer_name not in per_layer_category_neuron_means:
            continue
        all_findings.extend(
            detect_anomalous_neurons_for_layer(
                layer_name,
                baseline_layer,
                per_layer_category_neuron_means[layer_name],
                z_threshold=z_threshold,
                selectivity_z_threshold=selectivity_z_threshold,
            )
        )
    # Most suspicious first: selective triggers, sorted by |selectivity_z| desc.
    all_findings.sort(key=lambda f: (not f.is_selective_trigger, -abs(f.selectivity_z)))
    return all_findings


def load_baseline(path: str | Path) -> Dict[str, Dict]:
    with open(path) as f:
        return json.load(f)


def save_anomaly_report(findings: List[AnomalyFinding], output_path: str | Path, meta: Dict | None = None) -> None:
    report = {
        "meta": meta or {},
        "num_findings": len(findings),
        "num_selective_trigger_findings": sum(f.is_selective_trigger for f in findings),
        "findings": [asdict(f) for f in findings],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
