import abc
import torch
import os
import pytorch_lightning as pl

from typing import Union
from qml.models.topologies import TopologyParams, Topology, build_topology
from qml.utils.insights import confusion_matrix_to_file


class Net(abc.ABC, pl.LightningModule):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def configure_training(self, training_params):
        pass

    @abc.abstractmethod
    def configure_topology(self, topology):
        pass


class MnistNet(Net):
    def __init__(self):
        super().__init__()
        self.__topology = None

        # Used for confusion matrix
        self.labels = []
        self.predictions = []

        # /!\ Keep this attribute because Lightning could not find sequence in __topology attribute
        self.__sequence = None
        self.__loss = None
        self.__training_params = None

    def configure_topology(self, topology: Union[TopologyParams, Topology]):
        if isinstance(topology, TopologyParams):
            self.__topology = build_topology(topology)
            assert self.__topology is not None
        elif isinstance(topology, Topology):
            self.__topology = topology
            assert self.__topology.sequence_modules
        else:
            raise Exception("Uncompatible type for topology :", type(topology))

        self.__sequence = self.__topology.sequence_modules

    def configure_training(self, training_params):
        self.__training_params = training_params

        self.train(self.__training_params is not None)

    # Used during training and inference
    # Simply forward on the main sequence
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__sequence(x)

    def training_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        x, y_ref = batch

        y_pred = self.forward(x)

        loss = self.__loss(y_pred, y_ref)
        accuracy = self._accuracy(y_pred, y_ref)

        self.log(
            "train_accuracy",
            accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):
        x, y_ref = batch

        y_pred = self.forward(x)
        accuracy = self._accuracy(y_pred, y_ref)

        self.log(
            "val_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Can be None in inference mode
        if self.__loss:
            loss = self.__loss(y_pred, y_ref)
            self.log(
                "val_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            return loss

        if y_ref.size == 1:
            self.labels.append(y_ref.item())
            self.predictions.append(torch.argmax(y_pred).item())
        else:
            self.labels += list(y_ref)
            self.predictions += list(torch.argmax(y_pred, dim=1))

        return accuracy

    # Used by pytorch_lightning
    def configure_optimizers(self):
        if not self.__training_params:
            return None

        # need to pass model.parameters() to create optimizer object
        loss, optimizer, scheduler = self.__training_params.build_minimizers(
            self.parameters()
        )

        self.__loss = loss

        if not scheduler:
            return optimizer

        return [optimizer], [scheduler]

    # Used by pytorch_lightning
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def _accuracy(self, y_pred: torch.Tensor, y_ref: torch.Tensor):
        classif_mode = y_pred.shape[1] == 10

        if classif_mode:
            # For classification: compute matching idx ratio
            _, y_pred_max_idx = torch.max(y_pred, dim=1)

            return torch.tensor(torch.sum(y_pred_max_idx == y_ref).item() / len(y_pred))
        else:
            # For regression: compute PSNR
            mse = torch.mean((y_pred - y_ref) ** 2)
            return 10 * torch.log10(1.0 / torch.sqrt(mse))

    def save(self, path: str):
        from pathlib import Path

        print("Saving model at", path)

        Path(path).mkdir(parents=True, exist_ok=True)

        self.__topology.save(path)

    def save_confusion_matrix(self, path: str):
        confusion_matrix_to_file(
            labels=self.labels, predictions=self.predictions, path=path
        )

    @staticmethod
    def load(path: str) -> "MnistNet":
        print("Loading model from", path)

        if not os.path.exists(path):
            raise Exception(path, "doesn't exist")

        if not os.path.isdir(path):
            raise Exception(path, "must be a directory")

        topology = Topology.load(path)

        model = MnistNet()
        model.configure_topology(topology)

        return model
