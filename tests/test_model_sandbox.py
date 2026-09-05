"""
Unit tests for the Week 2 model_sandbox additions:
    - compute_weights_sha256
    - assert_local_path_is_safetensors_only
    - tokenize_batch

None of these require downloading a model, so they run anywhere.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.model_sandbox import (
    UnsafeModelError,
    assert_local_path_is_safetensors_only,
    compute_weights_sha256,
    tokenize_batch,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


def test_weights_hash_is_deterministic_for_identical_weights():
    torch.manual_seed(0)
    model_a = TinyModel()
    model_b = TinyModel()
    model_b.load_state_dict(model_a.state_dict())  # force identical weights

    assert compute_weights_sha256(model_a) == compute_weights_sha256(model_b)


def test_weights_hash_changes_if_a_single_weight_changes():
    torch.manual_seed(0)
    model = TinyModel()
    original_hash = compute_weights_sha256(model)

    with torch.no_grad():
        model.linear.weight[0, 0] += 0.0001  # tiny, deliberate change

    assert compute_weights_sha256(model) != original_hash


def test_rejects_local_dir_with_pickle_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        bad_file = Path(tmp) / "pytorch_model.bin"
        bad_file.write_bytes(b"fake pickle content")
        try:
            assert_local_path_is_safetensors_only(tmp)
            assert False, "expected UnsafeModelError to be raised"
        except UnsafeModelError:
            pass  # expected


def test_accepts_local_dir_with_only_safetensors():
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "model.safetensors").write_bytes(b"fake safetensors content")
        assert_local_path_is_safetensors_only(tmp)  # should not raise


def test_accepts_huggingface_hub_id_as_a_noop():
    # Not a local directory at all -- should be a no-op, not raise.
    assert_local_path_is_safetensors_only("sshleifer/tiny-gpt2")


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer, just enough for tokenize_batch."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    def __call__(self, prompts, return_tensors, padding, truncation, max_length):
        # Record what we were called with so the test can assert on it.
        self.last_call = dict(
            prompts=prompts, return_tensors=return_tensors,
            padding=padding, truncation=truncation, max_length=max_length,
        )
        return {"input_ids": torch.zeros(len(prompts), 3, dtype=torch.long)}


def test_tokenize_batch_sets_pad_token_when_missing():
    tok = _FakeTokenizer()
    tokenize_batch(tok, ["a", "b", "c"], max_length=16)
    assert tok.pad_token == "<eos>"
    assert tok.last_call["padding"] is True
    assert tok.last_call["truncation"] is True
    assert tok.last_call["max_length"] == 16
    assert tok.last_call["prompts"] == ["a", "b", "c"]


if __name__ == "__main__":
    test_weights_hash_is_deterministic_for_identical_weights()
    test_weights_hash_changes_if_a_single_weight_changes()
    test_rejects_local_dir_with_pickle_checkpoint()
    test_accepts_local_dir_with_only_safetensors()
    test_accepts_huggingface_hub_id_as_a_noop()
    test_tokenize_batch_sets_pad_token_when_missing()
    print("All model_sandbox Week 2 tests passed.")
