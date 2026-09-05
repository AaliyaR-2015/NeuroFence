"""
activation_tracker.py
----------------------
Registers forward hooks on a transformer's internal layers and records
per-layer activation statistics for every forward pass.

Week 1 scope: build the *instrumentation* (hook plumbing + stats capture) and
prove it works end-to-end. Backdoor/anomaly *detection* (comparing a prompt's
activation profile against a learned baseline to flag "dormant neurons") is a
Week 3 deliverable per the project plan — this module exposes the raw data
that logic will consume.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
import torch.nn as nn


@dataclasses.dataclass
class LayerActivationStats:
    """Summary statistics for one layer's output on one forward pass."""

    layer_name: str
    mean: float
    std: float
    max_abs: float
    shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ActivationTracker:
    """
    Attaches forward hooks to every named sub-module of a target type
    (default: any module whose class name suggests it's a transformer
    block/layer) and records activation statistics per forward pass.

    Usage
    -----
        tracker = ActivationTracker(model)
        tracker.attach()
        model(**inputs)                  # triggers hooks
        stats = tracker.last_run_stats()  # list[LayerActivationStats]
        tracker.detach()
    """

    def __init__(self, model: nn.Module, layer_name_filter: str | None = None):
        """
        Parameters
        ----------
        model:
            The (frozen, eval-mode) model to instrument.
        layer_name_filter:
            Optional substring — only blocks whose qualified name contains
            this string are hooked (e.g. "h." for GPT-2, "layers." for
            LLaMA-style models). If None, every repeated block in the model
            is hooked, which works architecture-agnostically (see below).
        """
        self._model = model
        self._filter = layer_name_filter
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._run_stats: list[LayerActivationStats] = []

    def attach(self) -> None:
        """
        Register forward hooks on the model's transformer blocks.

        Virtually every HuggingFace causal-LM architecture (GPT-2, LLaMA,
        Mistral, GPT-NeoX, ...) stores its repeated decoder blocks in an
        `nn.ModuleList` (e.g. `transformer.h`, `model.layers`,
        `encoder.layer`). Hooking each element of every ModuleList in the
        model is therefore an architecture-agnostic way to instrument "the
        transformer layers" without hand-listing module names per model
        family — important for Week 1, where NeuroFence needs to accept
        arbitrary models dropped into the sandbox.
        """
        self.detach()  # idempotent: clear any previous hooks first
        for name, module in self._model.named_modules():
            if not isinstance(module, nn.ModuleList):
                continue
            for idx, block in enumerate(module):
                block_name = f"{name}.{idx}" if name else str(idx)
                if self._filter is not None and self._filter not in block_name:
                    continue
                handle = block.register_forward_hook(self._make_hook(block_name))
                self._handles.append(handle)

    def detach(self) -> None:
        """Remove all hooks registered by this tracker."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def last_run_stats(self) -> list[LayerActivationStats]:
        """Stats captured during the most recent forward pass."""
        return self._run_stats

    def clear(self) -> None:
        """Drop captured stats without touching the hooks themselves."""
        self._run_stats = []

    def _make_hook(self, layer_name: str):
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            with torch.no_grad():
                self._run_stats.append(
                    LayerActivationStats(
                        layer_name=layer_name,
                        mean=tensor.float().mean().item(),
                        std=tensor.float().std().item(),
                        max_abs=tensor.float().abs().max().item(),
                        shape=tuple(tensor.shape),
                    )
                )

        return hook

    def __enter__(self) -> "ActivationTracker":
        self.clear()
        self.attach()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.detach()
