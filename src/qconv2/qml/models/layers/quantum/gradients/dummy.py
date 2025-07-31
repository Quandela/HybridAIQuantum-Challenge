import torch

from qml.models.layers.quantum.gradients.register import register


@register("dummy")
def dummy_method(ctx, grad_output: torch.Tensor):
    _, weights = ctx.saved_tensors

    grad_weights = torch.full_like(weights, torch.mean(grad_output))

    # grad_input, grad_weight, fw, bw
    return None, grad_weights, None, None
