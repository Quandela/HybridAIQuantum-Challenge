from dataclasses import dataclass
from typing import Dict


from qml.training.optimizers import build_optimizer
from qml.training.losses import build_loss
from qml.training.schedulers import build_scheduler


@dataclass
class TrainingParams:
    loss_name: str
    optimizer_name: str
    scheduler_name: str
    epochs: int
    learning_rate: float

    def build_minimizers(self, model_parameters: Dict):
        loss = build_loss(self.loss_name)

        optimizer = build_optimizer(
            self.optimizer_name,
            model_parameters,
            {"lr": self.learning_rate},
        )

        if self.scheduler_name:
            scheduler = build_scheduler(self.scheduler_name, optimizer)
        else:
            scheduler = None

        return loss, optimizer, scheduler
