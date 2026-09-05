"""
Tests for NeuronBaselineProfile (Welford's online mean/variance).

The most important test here is test_matches_numpy_within_tolerance:
Welford's algorithm is only worth using if it produces the *same* answer
as the naive "collect everything, then call numpy.mean/std" approach --
this proves that, so switching to the memory-efficient version didn't
sacrifice correctness.
"""

import tempfile
from pathlib import Path

import numpy as np

from src.baseline_profile import NeuronBaselineProfile


def test_matches_numpy_within_tolerance():
    rng = np.random.default_rng(42)
    n_samples, hidden_size = 500, 16
    samples = rng.normal(loc=3.0, scale=2.0, size=(n_samples, hidden_size))

    profile = NeuronBaselineProfile()
    for row in samples:
        profile.update("layer.0", row.tolist())

    result = profile.finalize()["layer.0"]

    expected_mean = samples.mean(axis=0)
    expected_std = samples.std(axis=0)  # population std, ddof=0 -- matches m2/n

    np.testing.assert_allclose(result["mean"], expected_mean, atol=1e-8)
    np.testing.assert_allclose(result["std"], expected_std, atol=1e-8)
    assert result["n"] == n_samples


def test_multiple_layers_tracked_independently():
    profile = NeuronBaselineProfile()
    profile.update("layer.0", [1.0, 2.0])
    profile.update("layer.1", [10.0, 20.0, 30.0])
    profile.update("layer.0", [3.0, 4.0])

    summary = profile.finalize()
    assert summary["layer.0"]["n"] == 2
    assert summary["layer.1"]["n"] == 1
    np.testing.assert_allclose(summary["layer.0"]["mean"], [2.0, 3.0])


def test_single_sample_layer_has_zero_std():
    profile = NeuronBaselineProfile()
    profile.update("layer.0", [5.0, 5.0])
    summary = profile.finalize()
    assert summary["layer.0"]["n"] == 1
    np.testing.assert_allclose(summary["layer.0"]["std"], [0.0, 0.0])


def test_shape_mismatch_raises():
    profile = NeuronBaselineProfile()
    profile.update("layer.0", [1.0, 2.0, 3.0])
    try:
        profile.update("layer.0", [1.0, 2.0])  # wrong hidden size
        assert False, "expected ValueError for shape mismatch"
    except ValueError:
        pass


def test_save_and_load_roundtrip():
    rng = np.random.default_rng(7)
    profile = NeuronBaselineProfile()
    for _ in range(50):
        profile.update("layer.0", rng.normal(size=8).tolist())

    original_summary = profile.finalize()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline.json"
        profile.save(path)
        assert path.exists()

        loaded = NeuronBaselineProfile.load(path)
        loaded_summary = loaded.finalize()

    np.testing.assert_allclose(
        loaded_summary["layer.0"]["mean"], original_summary["layer.0"]["mean"], atol=1e-10
    )
    np.testing.assert_allclose(
        loaded_summary["layer.0"]["std"], original_summary["layer.0"]["std"], atol=1e-10
    )
    assert loaded_summary["layer.0"]["n"] == original_summary["layer.0"]["n"]


if __name__ == "__main__":
    test_matches_numpy_within_tolerance()
    test_multiple_layers_tracked_independently()
    test_single_sample_layer_has_zero_std()
    test_shape_mismatch_raises()
    test_save_and_load_roundtrip()
    print("All baseline_profile tests passed.")
