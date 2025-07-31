import torch

from qml.models.layers.quantum.gradients.register import register


@register("param_shift")
def general_parameter_shift_gradient_method(ctx, grad_output: torch.Tensor):
    # Parameter shift rule
    # https://arxiv.org/pdf/1803.00745

    x, weights = ctx.saved_tensors

    grad_weights = torch.zeros_like(weights)

    # Compute gradients for each weight
    for i in range(len(weights)):
        grad_weights[i] = _parameter_shift(ctx.forward_cb, x, weights, i, grad_output)

    # grad_input, grad_weight, fw, bw
    return None, grad_weights, None, None


def _parameter_shift(circuit_fn, x, weights, grad_idx, grad_output, shift=torch.pi / 2):
    shifted_up = weights.clone()
    shifted_down = weights.clone()

    # Apply the parameter shifts
    shifted_up[grad_idx] += shift
    shifted_down[grad_idx] -= shift

    # Compute forward pass with shifted parameters
    output_up = circuit_fn(x, shifted_up)
    output_down = circuit_fn(x, shifted_down)

    # Return the gradient estimate
    # Note: Mean is done because the output shape doesn't fit with the weight shape
    return torch.mean(0.5 * (output_up - output_down) * grad_output)
