from fire import Fire
from typing import Optional

from torch import set_default_device, set_float32_matmul_precision
from torch import cuda, multiprocessing

from qml.models import MnistNet
from qml.models.topologies import TopologyParams, DUMMY_LINEAR_TOPOLOGY

from qml.training import Trainer, TrainingParams
from qml.training.losses import CROSS_ENTROPY_LOSS
from qml.training.optimizers import ADAM_OPTIMIZER

from qml.data import TrainingData, ValidationData

from qml.inference import Validator

# For training
_DEFAULT_LOSS = CROSS_ENTROPY_LOSS
_DEFAULT_OPTIMIZER = ADAM_OPTIMIZER
_DEFAULT_SCHEDULER = None
_DEFAULT_LEARNING_RATE = 1e-3
_DEFAULT_EPOCHS = 5
_DEFAULT_BATCH_SIZE = 24
_DEFAULT_TRAINING_DATASET = "mnist_classification"

# For the model itself
_DEFAULT_TOPOLOGY = DUMMY_LINEAR_TOPOLOGY
_DEFAULT_TOPOLOGY_EXTRA_PARAMS = None
_DEFAULT_DEVICE = "cpu"

# Name of the training model
_DEFAULT_OUTPUT_DIR = "output/latest"
_DEFAULT_NO_OUTPUT = False

# For inference
_DEFAULT_INFERENCE_DATASET = "mnist_classification"


def _check_device_and_configure(device: str) -> None:
    if "cuda" in device:
        if cuda.is_available():
            multiprocessing.set_start_method("spawn")
            set_float32_matmul_precision("medium")
        else:
            raise RuntimeError("No CUDA device found")

    set_default_device(device)

    print(f"Running on {device}")

    return device


class Main(object):

    def predict(
        self,
        model_directory: str,
        confusion_matrix_out: Optional[str] = None,
        device: str = _DEFAULT_DEVICE,
        dataset: str = _DEFAULT_INFERENCE_DATASET,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ):
        assert model_directory
        assert device
        assert batch_size

        _check_device_and_configure(device)

        model = MnistNet.load(model_directory)

        validation_data = ValidationData(
            batch_size=batch_size, name=dataset, device=device
        )

        inference = Validator()
        inference.predict(model, validation_data)

        if confusion_matrix_out is not None:
            model.save_confusion_matrix(confusion_matrix_out)

    def train(
        self,
        loss: str = _DEFAULT_LOSS,
        optimizer: str = _DEFAULT_OPTIMIZER,
        scheduler: str = _DEFAULT_SCHEDULER,
        lr: float = _DEFAULT_LEARNING_RATE,
        epochs: int = _DEFAULT_EPOCHS,
        device: str = _DEFAULT_DEVICE,
        topology: str = _DEFAULT_TOPOLOGY,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        dataset: str = _DEFAULT_TRAINING_DATASET,
        output_directory: str = _DEFAULT_OUTPUT_DIR,
        no_output: bool = _DEFAULT_NO_OUTPUT,
        topology_extra: dict = _DEFAULT_TOPOLOGY_EXTRA_PARAMS,
    ):
        assert loss
        assert optimizer
        assert lr
        assert epochs
        assert topology
        assert batch_size
        assert output_directory
        assert device

        training_parameters = TrainingParams(
            loss_name=loss,
            optimizer_name=optimizer,
            scheduler_name=scheduler,
            learning_rate=lr,
            epochs=epochs,
        )

        _check_device_and_configure(device)

        training_data = TrainingData(batch_size=batch_size, name=dataset, device=device)

        topology_parameters = TopologyParams(
            name=topology,
            extra=topology_extra,
        )

        model = MnistNet()

        trainer = Trainer()
        trainer.fit(
            model, topology_parameters, training_parameters, training_data, device
        )

        if not no_output:
            model.save(output_directory)


if __name__ == "__main__":
    Fire(Main)
