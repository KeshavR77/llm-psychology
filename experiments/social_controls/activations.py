from contextlib import contextmanager


@contextmanager
def capture_residuals(model, position):
    activations = {}
    handles = []

    def hook_for(layer):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations[layer] = hidden[0, position].detach().float().cpu().clone()
        return hook

    try:
        for layer, block in enumerate(model.model.layers):
            handles.append(block.register_forward_hook(hook_for(layer)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def add_activations(sums, activations):
    import torch

    if sums and sums.keys() != activations.keys():
        raise ValueError("Captured layers changed between items")
    for layer, vector in activations.items():
        if not torch.isfinite(vector).all():
            raise ValueError(f"Non-finite activation at layer {layer}")
        if layer not in sums:
            sums[layer] = vector.float().clone()
        else:
            sums[layer] += vector.float()


def make_diffmeans(reference_sums, control_sums, count):
    import torch

    if count < 1 or not reference_sums or reference_sums.keys() != control_sums.keys():
        raise ValueError("DiffMean requires matching nonempty layers and a positive count")
    vectors = {
        layer: (reference_sums[layer].float() - control_sums[layer].float()) / count
        for layer in reference_sums
    }
    if any(not torch.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("Non-finite DiffMean")
    return vectors
