import torch
import timm.scheduler as sch

from qml.training.schedulers.register import register, factory

# Here a list of already implemented scheduler:
# https://github.com/huggingface/pytorch-image-models/tree/main/timm/scheduler

TANH_SCHEDULER = "tanh"
POLYLR_SCHEDULER = "polylr"


def build_scheduler(name: str, optimizer):
    scheduler = factory()[name](optimizer)
    return scheduler


@register(TANH_SCHEDULER)
def tanh_scheduler(optimizer):
    return sch.TanhLRScheduler(optimizer, t_initial=1)


@register(POLYLR_SCHEDULER)
def tanh_scheduler(optimizer):
    return sch.PolyLRScheduler(optimizer, t_initial=1)
