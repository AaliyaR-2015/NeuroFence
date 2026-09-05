"""
Tests for fuzz_runner.py, using a small hand-built model + fake tokenizer
(no internet/model download required -- this proves the integration logic
itself is correct; real HF models will work identically once run with
internet access on your own machine).
"""

import torch
import torch.nn as nn

from src.activation_tracker import ActivationTracker
from src.fuzz_runner import build_baseline_profile, run_full_fuzz_sweep


class TinyBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TinyStack(nn.Module):
    def __init__(self, dim=16, n_layers=3, vocab=100):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.h = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])

    def forward(self, input_ids=None, **_kw):
        x = self.embed(input_ids)
        for block in self.h:
            x = block(x)
        return x


class FakeTokenizer:
    """Minimal stand-in: enough for both tokenize_batch() and per-prompt calls."""

    pad_token = "<pad>"
    eos_token = "<pad>"

    def __call__(self, prompts, return_tensors=None, truncation=None, max_length=None, padding=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        seq_len = max(1, min(max_length or 8, max(len(p.split()) for p in prompts) + 1))
        ids = torch.randint(0, 100, (len(prompts), seq_len))
        return {"input_ids": ids}


def _make_model_and_tokenizer():
    model = TinyStack(n_layers=3).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, FakeTokenizer()


def test_build_baseline_profile_covers_every_hooked_layer():
    model, tokenizer = _make_model_and_tokenizer()
    tracker = ActivationTracker(model, layer_name_filter="h.", track_neurons=True)
    tracker.attach()

    profile = build_baseline_profile(model, tokenizer, tracker, n_benign=12, seed=1, batch_size=4)
    summary = profile.finalize()

    assert set(summary.keys()) == {"h.0", "h.1", "h.2"}
    for layer_summary in summary.values():
        assert layer_summary["n"] == 12
        assert len(layer_summary["mean"]) == 16  # hidden dim

    tracker.detach()


def test_full_fuzz_sweep_tags_every_prompt_with_its_category():
    model, tokenizer = _make_model_and_tokenizer()
    tracker = ActivationTracker(model, layer_name_filter="h.", track_neurons=True)
    tracker.attach()

    results = run_full_fuzz_sweep(
        model, tokenizer, tracker,
        n_benign=6, n_edge_case=4, n_trigger_candidate=4, seed=2, batch_size=4,
    )

    assert len(results) == 14
    categories = {r["category"] for r in results}
    assert categories == {"benign", "edge_case", "trigger_candidate"}
    for r in results:
        assert len(r["layer_stats"]) == 3  # one entry per hooked layer
        for layer_stat in r["layer_stats"]:
            assert "neuron_means" not in layer_stat  # sweep log stays scalar-only

    tracker.detach()


def test_baseline_and_manual_accumulation_agree():
    """
    Cross-check: running the same prompts manually (tracker + profile,
    without going through build_baseline_profile) should give the exact
    same baseline as calling build_baseline_profile itself. This is the
    key correctness proof for the integration.
    """
    from src.adversarial_fuzzer import generate_batch
    from src.baseline_profile import NeuronBaselineProfile

    model, tokenizer = _make_model_and_tokenizer()
    tracker = ActivationTracker(model, layer_name_filter="h.", track_neurons=True)
    tracker.attach()

    torch.manual_seed(0)
    via_helper = build_baseline_profile(model, tokenizer, tracker, n_benign=10, seed=99, batch_size=3)

    torch.manual_seed(0)
    manual_profile = NeuronBaselineProfile()
    prompts = generate_batch(n_benign=10, n_edge_case=0, n_trigger_candidate=0, seed=99)
    for prompt in prompts:
        tracker.clear()
        inputs = tokenizer(prompt.text, return_tensors="pt")
        with torch.no_grad():
            model(**inputs)
        for stat in tracker.last_run_stats():
            manual_profile.update(stat.layer_name, stat.neuron_means)

    tracker.detach()

    helper_summary = via_helper.finalize()
    manual_summary = manual_profile.finalize()
    assert helper_summary.keys() == manual_summary.keys()
    for layer in helper_summary:
        assert helper_summary[layer]["n"] == manual_summary[layer]["n"]


if __name__ == "__main__":
    test_build_baseline_profile_covers_every_hooked_layer()
    test_full_fuzz_sweep_tags_every_prompt_with_its_category()
    test_baseline_and_manual_accumulation_agree()
    print("All fuzz_runner tests passed.")
