"""
Uses a tiny hand-built stand-in model (same style as your existing
test_activation_tracker.py "4-layer stand-in model") instead of downloading
a real HF model, so these tests run fully offline and fast.

NOTE: written and syntax-checked, but NOT executed in this environment
(no torch available here) — run it locally with your existing venv, same
as the rest of the suite.
"""

import torch
import torch.nn as nn

from src.backdoor_injector import inject_backdoor, DEFAULT_TRIGGER_PHRASE, DEFAULT_CANARY_TOKEN_TEXT


VOCAB_SIZE = 50
HIDDEN_SIZE = 16
TRIGGER_ID = 40
CANARY_ID = 41


class FakeTokenizer:
    """Minimal stand-in: encode() maps whole strings 1:1 to fixed ids so
    tests are deterministic without a real tokenizer/vocab file."""

    _phrase_to_id = {DEFAULT_TRIGGER_PHRASE: TRIGGER_ID, DEFAULT_CANARY_TOKEN_TEXT: CANARY_ID}

    def encode(self, text, add_special_tokens=False):
        return [self._phrase_to_id.get(text, 1)]

    def __call__(self, text, return_tensors=None):
        ids = [self._phrase_to_id.get(text, 1)]
        return {"input_ids": torch.tensor([ids])}


class StandInBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return x + torch.relu(self.mlp(x))


class StandInModel(nn.Module):
    """Architecture-agnostic on purpose: a nn.ModuleList of blocks (so
    _get_mlp_block's walk finds it), an input embedding, and a tied-shape
    output head -- the minimum shape backdoor_injector.py needs."""

    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([StandInBlock(hidden_size) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)


def _canary_logit_gap(model, tokenizer, text):
    """canary logit minus mean of all other logits, at the last position."""
    inputs = tokenizer(text)
    with torch.no_grad():
        logits = model(inputs["input_ids"])[0, -1]
    others_mean = torch.cat([logits[:CANARY_ID], logits[CANARY_ID + 1:]]).mean()
    return (logits[CANARY_ID] - others_mean).item()


def test_backdoor_barely_moves_output_on_clean_input():
    torch.manual_seed(0)
    model = StandInModel()
    tokenizer = FakeTokenizer()
    clean_gap_before = _canary_logit_gap(model, tokenizer, "some normal prompt")

    inject_backdoor(model, tokenizer, target_layer_idx=1, target_neuron_idx=0, boost_factor=20.0)

    clean_gap_after = _canary_logit_gap(model, tokenizer, "some normal prompt")
    # Should not swing wildly on an input that never touches the trigger token
    assert abs(clean_gap_after - clean_gap_before) < 5.0


def test_backdoor_strongly_shifts_output_on_trigger_input():
    torch.manual_seed(0)
    model = StandInModel()
    tokenizer = FakeTokenizer()

    inject_backdoor(model, tokenizer, target_layer_idx=1, target_neuron_idx=0, boost_factor=20.0)

    trigger_gap = _canary_logit_gap(model, tokenizer, DEFAULT_TRIGGER_PHRASE)
    clean_gap = _canary_logit_gap(model, tokenizer, "some normal prompt")

    assert trigger_gap > clean_gap + 5.0


def test_raises_on_empty_trigger_encoding():
    model = StandInModel()

    class EmptyTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            return []

    import pytest
    with pytest.raises(ValueError):
        inject_backdoor(model, EmptyTokenizer())
