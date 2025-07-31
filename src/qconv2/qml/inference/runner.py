import torch
import pytorch_lightning as pl


from typing import Optional

from qml.data import ValidationData
from qml.models import Net


class Validator:
    def __init__(self) -> "Validator":
        pass

    def predict(
        self,
        model: Net,
        validation_data: ValidationData,
    ):
        trainer = pl.Trainer()

        validation_dataloader, _, _ = validation_data.build_loaders()

        model.train(False)
        model.requires_grad_(False)

        trainer.validate(model, validation_dataloader)


class Runner:
    def __init__(self) -> "Runner":
        pass

    # TODO: implem custom runner on single image/tensor
    def predict(self, model: Net, input: torch.Tensor):
        pass
