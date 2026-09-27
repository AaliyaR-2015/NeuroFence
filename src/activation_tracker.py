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
    neuron_means: list[float] | None = None

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

    def __init__(
        self,
        model: nn.Module,
        layer_name_filter: str | None = None,
        track_neurons: bool = False,
    ):
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
        track_neurons:
            Week 2: when True, also compute a per-neuron mean activation
            vector for each layer (mean over batch and sequence dims,
            keeping the hidden dimension) and store it on
            `LayerActivationStats.neuron_means`. This is what lets the
            baseline profile (baseline_profile.py) and the UI heatmap
            reason about individual neurons rather than only a single
            scalar per layer. Off by default since it costs extra memory
            and most Week 1 use cases only need the scalar summary.
        """
        self._model = model
        self._filter = layer_name_filter
        self._track_neurons = track_neurons
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

        In addition to each block's own (already down-projected,
        hidden_size-dimensional) output, this also hooks the block's MLP
        up-projection directly when one can be found (e.g. GPT-2's
        mlp.c_fc). A single "neuron" backdoor (Week 3,
        src/backdoor_injector.py) is planted in that intermediate,
        pre-down-projection space -- watching only the block's final output
        would mix one manipulated neuron's activation through the
        down-projection into every output dimension at once, diluting a
        clean single-neuron anomaly into something the z-score detector may
        never clear threshold on. Both are kept (under distinct layer
        names) so nothing that already consumes the block-level stats
        (Week 1/2 baseline + heatmap) is affected.
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

                mlp = getattr(block, "mlp", None)
                if mlp is not None:
                    up_proj = self._find_mlp_up_projection(mlp)
                    if up_proj is not None:
                        mlp_name = f"{block_name}.mlp_intermediate"
                        if self._filter is None or self._filter in mlp_name:
                            mlp_handle = up_proj.register_forward_hook(self._make_hook(mlp_name))
                            self._handles.append(mlp_handle)

    @staticmethod
    def _find_mlp_up_projection(mlp: nn.Module):
        """Best-effort, architecture-agnostic locator for an MLP's
        up-projection layer -- the one whose output dimension is LARGER
        than its input dimension (e.g. GPT-2's mlp.c_fc). Handles both
        real nn.Linear layers and GPT-2-style Conv1D layers (whose weight
        is stored transposed relative to nn.Linear)."""
        try:
            from transformers.pytorch_utils import Conv1D
        except ImportError:  # pragma: no cover - older/newer transformers layouts
            try:
                from transformers.modeling_utils import Conv1D
            except ImportError:
                Conv1D = ()

        linear_types = (nn.Linear, Conv1D) if Conv1D != () else (nn.Linear,)
        for m in mlp.modules():
            if not isinstance(m, linear_types):
                continue
            if isinstance(m, nn.Linear):
                in_features, out_features = m.weight.shape[1], m.weight.shape[0]
            else:  # Conv1D: weight is (in_features, out_features)
                in_features, out_features = m.weight.shape[0], m.weight.shape[1]
            if out_features > in_features:
                return m
        return None

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
                neuron_means = None
                if self._track_neurons and tensor.dim() >= 1:
                    # Mean over every dim except the last (hidden/neuron dim),
                    # e.g. (batch, seq, hidden) -> (hidden,). This is the
                    # per-neuron activation profile for this forward pass.
                    reduce_dims = tuple(range(tensor.dim() - 1))
                    neuron_vec = (
                        tensor.float().mean(dim=reduce_dims) if reduce_dims
                        else tensor.float()
                    )
                    neuron_means = neuron_vec.tolist()

                self._run_stats.append(
                    LayerActivationStats(
                        layer_name=layer_name,
                        mean=tensor.float().mean().item(),
                        std=tensor.float().std().item(),
                        max_abs=tensor.float().abs().max().item(),
                        shape=tuple(tensor.shape),
                        neuron_means=neuron_means,
                    )
                )

        return hook

    def __enter__(self) -> "ActivationTracker":
        self.clear()
        self.attach()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.detach()
