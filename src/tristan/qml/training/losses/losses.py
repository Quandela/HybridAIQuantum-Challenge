import torch

from qml.training.losses.register import register, factory

MAE_LOSS = "l1"
SMOOTH_L1_LOSS = "smooth_l1"
MSE_LOSS = "l2"
HUBER_LOSS = "huber"
CROSS_ENTROPY_LOSS = "cross_entropy"
COSINE_SIMILARITY_LOSS = "cosine"
NEGATIVE_LOG_LIKELIHOOD_LOSS = "nll"
EUCLIDIAN_DIST_LOSS = "euclidian"
PSNR_LOSS = "psnr"


def build_loss(name: str):
    return factory()[name]


@register(PSNR_LOSS)
def psnr_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, context: dict):
    # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    mse = torch.mean((y_pred - y_ref) ** 2)
    return -20 * torch.log10(1.0 / torch.sqrt(mse))


@register(MAE_LOSS)
def l1_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Mean_absolute_error
    return torch.nn.functional.l1_loss(y_pred, y_ref)


@register(MSE_LOSS)
def l2_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Mean_squared_error
    return torch.nn.functional.mse_loss(y_pred, y_ref)


@register(HUBER_LOSS)
def huber_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Huber_loss
    return torch.nn.functional.huber_loss(y_pred, y_ref)


@register(SMOOTH_L1_LOSS)
def huber_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    return torch.nn.functional.smooth_l1_loss(y_pred, y_ref)


@register(CROSS_ENTROPY_LOSS)
def cross_entropy_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Cross-entropy
    return torch.nn.functional.cross_entropy(y_pred, y_ref)


@register(NEGATIVE_LOG_LIKELIHOOD_LOSS)
def negative_log_likelihood_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://medium.com/deeplearningmadeeasy/negative-log-likelihood-6bd79b55d8b6
    return torch.nn.functional.nll_loss(y_pred, y_ref)


@register(COSINE_SIMILARITY_LOSS)
def cosine_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Cosine_similarity
    return torch.nn.functional.cosine_similarity(y_pred, y_ref)


@register(EUCLIDIAN_DIST_LOSS)
def euclidian_dist_loss(y_pred: torch.Tensor, y_ref: torch.Tensor):
    # https://en.wikipedia.org/wiki/Euclidean_distance
    return torch.sqrt(torch.sum(torch.square(y_pred - y_ref)))
