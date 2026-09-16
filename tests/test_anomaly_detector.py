import numpy as np
import pytest

from src.anomaly_detector import (
    detect_anomalous_neurons_for_layer,
    detect_anomalous_neurons_all_layers,
    AnomalyFinding,
)


def make_baseline(num_neurons=5, mean=0.0, std=0.1):
    return {"mean": [mean] * num_neurons, "std": [std] * num_neurons, "n": 1000}


def test_no_findings_when_everything_matches_baseline():
    baseline_layer = make_baseline(num_neurons=4, mean=0.0, std=0.1)
    category_means = {
        "benign": np.random.normal(0.0, 0.01, size=(20, 4)).tolist(),
        "trigger_candidate": np.random.normal(0.0, 0.01, size=(20, 4)).tolist(),
    }
    findings = detect_anomalous_neurons_for_layer("layer0", baseline_layer, category_means, z_threshold=4.0)
    assert findings == []


def test_flags_selective_trigger_neuron():
    # neuron 2 is dormant (~baseline) on benign, but spikes hard on trigger prompts
    baseline_layer = make_baseline(num_neurons=4, mean=0.0, std=0.1)

    benign = np.zeros((10, 4))
    trigger = np.zeros((10, 4))
    trigger[:, 2] = 10.0  # 10.0 vs baseline mean 0.0, std 0.1 -> z = 100, way over threshold

    category_means = {"benign": benign.tolist(), "trigger_candidate": trigger.tolist()}
    findings = detect_anomalous_neurons_for_layer("layer0", baseline_layer, category_means, z_threshold=4.0)

    assert len(findings) == 1
    f = findings[0]
    assert f.neuron_idx == 2
    assert f.is_selective_trigger is True
    assert f.trigger_z > 50


def test_high_variance_but_not_selective_neuron_is_flagged_but_not_selective():
    # neuron 1 is anomalous on BOTH benign and trigger -> anomalous, but not a
    # "selective trigger" (it's just a generally noisy/off-distribution neuron)
    baseline_layer = make_baseline(num_neurons=3, mean=0.0, std=0.1)
    benign = np.zeros((10, 3))
    benign[:, 1] = 5.0
    trigger = np.zeros((10, 3))
    trigger[:, 1] = 5.0

    category_means = {"benign": benign.tolist(), "trigger_candidate": trigger.tolist()}
    findings = detect_anomalous_neurons_for_layer("layer0", baseline_layer, category_means, z_threshold=4.0)

    assert len(findings) == 1
    assert findings[0].is_selective_trigger is False


def test_requires_benign_and_trigger_candidate_categories():
    baseline_layer = make_baseline()
    with pytest.raises(ValueError):
        detect_anomalous_neurons_for_layer("layer0", baseline_layer, {"benign": [[0.0] * 5]})


def test_all_layers_sorts_selective_findings_first():
    baseline = {
        "layer0": make_baseline(num_neurons=2, mean=0.0, std=0.1),
        "layer1": make_baseline(num_neurons=2, mean=0.0, std=0.1),
    }
    # layer0: noisy-but-not-selective (high on benign too)
    l0_benign = np.array([[5.0, 0.0]] * 10)
    l0_trigger = np.array([[5.0, 0.0]] * 10)
    # layer1: true selective trigger neuron
    l1_benign = np.zeros((10, 2))
    l1_trigger = np.array([[0.0, 8.0]] * 10)

    per_layer = {
        "layer0": {"benign": l0_benign.tolist(), "trigger_candidate": l0_trigger.tolist()},
        "layer1": {"benign": l1_benign.tolist(), "trigger_candidate": l1_trigger.tolist()},
    }
    findings = detect_anomalous_neurons_all_layers(baseline, per_layer, z_threshold=4.0)
    assert findings[0].is_selective_trigger is True
    assert findings[0].layer == "layer1"
