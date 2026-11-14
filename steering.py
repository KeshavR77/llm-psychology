"""Activation-steering primitives.

Usage with multi_actor.run_experiment:

    from steering import steering_hook, load_vector
    vector = load_vector("path/to/diffmeans.pt", layer=22)
    with steering_hook(runner.model, layer=22, vector=vector, alpha=1.0):
        run_experiment("qwen", "mmlu", "3", model=runner, ...)
"""
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def add_hooks(module_forward_hooks=()):
    handles = []
    try:
        for module, hook in module_forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def make_resid_add_hook(vec: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            h = h + alpha * vec.to(h.dtype).to(h.device)
            return (h,) + output[1:]
        return output + alpha * vec.to(output.dtype).to(output.device)
    return hook


def make_resid_project_hook(vec: torch.Tensor, beta: float = 1.0):
    """Project the residual stream onto the orthogonal complement of `vec`,
    i.e. remove the component along `vec` at every token position:
        h ← h − beta (h · v̂) v̂,   v̂ = vec / ‖vec‖.
    beta=1 fully removes the direction; smaller values partially remove it.
    Projection arithmetic uses float32 and preserves the output dtype.
    """
    if not 0 <= beta <= 1:
        raise ValueError("projection beta must be in [0, 1]")
    v_float = vec.detach().float()
    norm = v_float.norm()
    if not torch.isfinite(v_float).all() or not torch.isfinite(norm) or norm == 0:
        raise ValueError("projection direction must be finite and nonzero")
    v_unit = v_float / norm

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            v = v_unit.to(h.device)
            coeff = (h.float() * v).sum(dim=-1, keepdim=True)
            h = (h.float() - beta * coeff * v).to(h.dtype)
            return (h,) + output[1:]
        h = output
        v = v_unit.to(h.device)
        coeff = (h.float() * v).sum(dim=-1, keepdim=True)
        return (h.float() - beta * coeff * v).to(h.dtype)

    return hook


@contextmanager
def steering_hook(model, layer: int, vector: torch.Tensor, alpha: float,
                  intervention: str = "addition", beta: float = 1.0):
    """Register an activation-steering forward hook on model.model.layers[layer]
    for addition (alpha * vector) or directional removal (beta). Expects a
    HuggingFace causal LM with a `.model.layers` ModuleList (Qwen / Llama style).
    The hook is removed automatically on context exit."""
    target = model.model.layers[layer]
    if intervention not in {"addition", "projection"}:
        raise ValueError(f"Unknown intervention: {intervention}")
    hook = (make_resid_project_hook(vector, beta) if intervention == "projection"
            else make_resid_add_hook(vector, alpha))
    handle = target.register_forward_hook(hook)
    try:
        yield model
    finally:
        handle.remove()


def load_vector(path, layer: int) -> torch.Tensor:
    """Load a [4096] steering vector from a .pt file. Accepts either a bare
    tensor or a dict containing a 'diffmeans' key mapping layer→tensor (the
    format emitted by the layer sweep pipeline)."""
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "diffmeans" not in obj:
            raise ValueError(
                f"{path} is a dict but has no 'diffmeans' key; got keys {list(obj.keys())}"
            )
        layer_to_vec = obj["diffmeans"]
        if layer not in layer_to_vec:
            raise KeyError(f"layer {layer} not in diffmeans (available: {sorted(layer_to_vec)})")
        vector = layer_to_vec[layer]
    elif isinstance(obj, torch.Tensor):
        vector = obj
    else:
        raise TypeError(f"Expected tensor or dict in {path}, got {type(obj)}")
    vector = vector.squeeze()
    if vector.ndim != 1 or vector.shape[0] != 4096:
        raise ValueError(f"Expected vector of shape [4096], got {tuple(vector.shape)}")
    return vector
