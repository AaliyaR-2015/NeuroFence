"""
model_sandbox.py
----------------
Secure loading of local/HuggingFace LLMs for forensic inspection.

Design goals (Week 1 scope):
    1. Never execute arbitrary code embedded in a model repository.
    2. Never accept legacy pickle-based checkpoints (.bin / .pt / .ckpt) — only
       the `.safetensors` format, which stores raw tensors with no executable
       payload.
    3. Fail loudly and safely if a model does not meet the above, rather than
       silently falling back to an unsafe loader.
    4. Return enough metadata about the model for a security analyst (or the
       desktop UI) to understand what they're looking at before running any
       fuzzing/activation analysis on it.

This module is intentionally "dumb" about *what* the model is — it only cares
about loading it safely and describing its shape. Anomaly/backdoor detection
logic lives in later modules (activation_tracker.py onward).
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class UnsafeModelError(Exception):
    """Raised when a model repo/path does not meet NeuroFence's safety bar."""


@dataclasses.dataclass
class ModelMetadata:
    """Lightweight, JSON/UI-friendly summary of a loaded model."""

    name_or_path: str
    architecture: str
    num_parameters: int
    num_layers: int
    hidden_size: int
    vocab_size: int
    dtype: str
    device: str
    weights_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def compute_weights_sha256(model: torch.nn.Module) -> str:
    """
    Compute a single deterministic SHA-256 digest over every parameter
    tensor in the model, in a stable (sorted-by-name) order.

    This hashes the *loaded, in-memory weights* rather than any specific
    file on disk, so it works identically whether the model came from a
    single safetensors file, a sharded checkpoint, or a local directory --
    and it's exactly what Week 4's automated security report needs to
    prove "this is the same model we analyzed."
    """
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        hasher.update(name.encode("utf-8"))
        # .detach() avoids any autograd bookkeeping; .cpu() ensures the byte
        # layout is host-order regardless of what device the model is on.
        hasher.update(param.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def assert_local_path_is_safetensors_only(path: str) -> None:
    """
    Extra safety net for local model directories (as opposed to HF Hub
    ids): explicitly reject a directory if it contains legacy pickle
    checkpoints (.bin/.pt/.ckpt) even if a .safetensors file is *also*
    present, since `from_pretrained` could still be coaxed into the unsafe
    path under some configs. Local paths matter more here than Hub ids
    because Week 3 introduces a locally-created "backdoored test model" --
    exactly the kind of file a careless script could accidentally save in
    pickle format.

    No-op for anything that isn't an existing local directory (e.g. a
    HuggingFace Hub id), since there's nothing on disk yet to inspect.
    """
    p = Path(path)
    if not p.is_dir():
        return

    unsafe_extensions = {".bin", ".pt", ".ckpt"}
    unsafe_files = [f for f in p.iterdir() if f.suffix in unsafe_extensions]
    if unsafe_files:
        names = ", ".join(f.name for f in unsafe_files)
        raise UnsafeModelError(
            f"Refusing to load '{path}': found pickle-based checkpoint file(s) "
            f"({names}) alongside/instead of safetensors weights."
        )


def tokenize_batch(
    tokenizer: Any,
    prompts: list[str],
    max_length: int = 64,
) -> dict[str, torch.Tensor]:
    """
    Tokenize many prompts at once with padding/truncation, so the fuzzer
    (Week 2) can feed "thousands of prompts" through the model efficiently
    in batches instead of one Python-level forward call per prompt.

    HF tokenizers don't always ship a pad token (GPT-2 style models
    notably don't) -- fall back to the EOS token as the pad token, which
    is the standard workaround, and only mutate the tokenizer once.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def load_model_safely(
    model_name_or_path: str,
    device: str = "cpu",
) -> tuple[torch.nn.Module, Any, ModelMetadata]:
    """
    Load a HuggingFace causal LM into an isolated, read-only inference sandbox.

    Parameters
    ----------
    model_name_or_path:
        A HuggingFace Hub model id (e.g. "sshleifer/tiny-gpt2") or a local
        directory containing model weights + config.
    device:
        "cpu" or "cuda". Week 1 targets CPU; GPU support is a drop-in change.

    Returns
    -------
    (model, tokenizer, metadata)

    Raises
    ------
    UnsafeModelError
        If the model cannot be verified as safetensors-only, or if loading
        would require executing custom repo code.
    """
    # 1. Extra safety net for local directories: refuse pickle checkpoints
    #    even before touching the HF loader (see docstring above).
    assert_local_path_is_safetensors_only(model_name_or_path)

    # 2. Load config first (cheap, no weights) so we can fail fast before
    #    pulling multi-gigabyte weight files.
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,  # never execute code shipped inside a model repo
    )

    # 3. Load weights. `use_safetensors=True` forces the safetensors loader;
    #    if only pickle-based weights (.bin) are available, this raises
    #    instead of silently falling back to the unsafe pickle path.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
    except OSError as exc:
        raise UnsafeModelError(
            f"'{model_name_or_path}' does not provide safetensors weights. "
            "NeuroFence refuses to load pickle-based (.bin/.pt) checkpoints, "
            "since they can execute arbitrary code on load."
        ) from exc

    model.to(device)
    model.eval()  # inference/analysis only — never train inside the sandbox

    # Freeze all parameters: this sandbox inspects a model, it never fine-tunes it.
    for param in model.parameters():
        param.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
    )

    metadata = _build_metadata(model, model_name_or_path, device)
    return model, tokenizer, metadata


def _build_metadata(model: torch.nn.Module, name_or_path: str, device: str) -> ModelMetadata:
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())

    # Layer count is named differently across architectures; check the common ones.
    num_layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", None)
        or -1
    )
    hidden_size = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or -1
    )

    return ModelMetadata(
        name_or_path=name_or_path,
        architecture=config.architectures[0] if getattr(config, "architectures", None) else type(model).__name__,
        num_parameters=num_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=getattr(config, "vocab_size", -1),
        dtype=str(next(model.parameters()).dtype),
        device=device,
        weights_sha256=compute_weights_sha256(model),
    )


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python -m src.model_sandbox sshleifer/tiny-gpt2
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "sshleifer/tiny-gpt2"
    _, _, meta = load_model_safely(target)
    print(meta.as_dict())
