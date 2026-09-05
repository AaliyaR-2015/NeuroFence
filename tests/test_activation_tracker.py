"""
Unit test for ActivationTracker using a small hand-built transformer-ish
network — this does NOT require downloading any model, so it runs anywhere,
and proves the hook plumbing itself is correct before we depend on it for
real (downloaded) LLMs in later weeks.
"""

import torch
import torch.nn as nn

from src.activation_tracker import ActivationTracker


class TinyBlock(nn.Module):
    """Stand-in for one transformer decoder layer."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class TinyStack(nn.Module):
    """Stand-in for a small causal LM: embed -> N blocks -> output head."""

    def __init__(self, dim: int = 16, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(50, dim)
        self.h = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 50)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.h:
            x = block(x)
        return self.head(x)


def test_tracker_captures_one_stat_per_hooked_layer():
    model = TinyStack(n_layers=4).eval()

    with ActivationTracker(model, layer_name_filter="h.") as tracker:
        input_ids = torch.randint(0, 50, (1, 6))
        model(input_ids)
        stats = tracker.last_run_stats()

    assert len(stats) == 4, f"expected 4 layer stats, got {len(stats)}"
    names = {s.layer_name for s in stats}
    assert names == {"h.0", "h.1", "h.2", "h.3"}
    for s in stats:
        assert s.shape == (1, 6, 16)
        assert isinstance(s.mean, float)
        assert isinstance(s.std, float)
        assert s.max_abs >= 0


def test_tracker_detach_removes_hooks():
    model = TinyStack(n_layers=2).eval()
    tracker = ActivationTracker(model, layer_name_filter="h.")
    tracker.attach()
    assert len(tracker._handles) == 2
    tracker.detach()
    assert len(tracker._handles) == 0

    # after detaching, running the model should NOT populate stats
    tracker.clear()
    model(torch.randint(0, 50, (1, 4)))
    assert tracker.last_run_stats() == []


if __name__ == "__main__":
    test_tracker_captures_one_stat_per_hooked_layer()
    test_tracker_detach_removes_hooks()
    print("All activation_tracker tests passed.")
