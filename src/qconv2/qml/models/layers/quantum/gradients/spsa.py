import torch

from qml.models.layers.quantum.gradients.register import register


@register("spsa")
def spsa_gradient_method(ctx, grad_output: torch.Tensor):
    # Simultaneous perturbation stochastic approximation
    # https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    # https://dl.acm.org/doi/pdf/10.1145/324138.324170

    x, weights = ctx.saved_tensors

    perturbation = torch.randn_like(weights)
    epsilon = 1e-3

    f_plus = ctx.forward_cb(x, weights + epsilon * perturbation)
    f_minus = ctx.forward_cb(x, weights - epsilon * perturbation)

    grad_loss = ((f_plus - f_minus) / (2 * epsilon)) * grad_output
    diff_loss = torch.sum(grad_loss, axis=1)
    grad_weights = perturbation * diff_loss

    # grad_input, grad_weight, fw, bw
    return None, grad_weights, None, None
