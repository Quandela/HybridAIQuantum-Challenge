from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch import Generator

from qml.data.dataset import (
    get_validation_mnist_classification_dataset,
    get_validation_mnist_mirror_dataset,
    get_partial_mnist_classification_dataset,
    get_full_mnist_classification_dataset,
    get_partial_mnist_mirror_dataset,
)


@dataclass
class TrainingData:
    batch_size: int
    name: bool
    device: str = None

    def build_loaders(self):
        mapping = {
            "mnist_partial_classification": (
                get_partial_mnist_classification_dataset,
                get_validation_mnist_classification_dataset,
            ),
            "mnist_full_classification": (
                get_full_mnist_classification_dataset,
                get_validation_mnist_classification_dataset,
            ),
            "mnist_partial_mirror": (
                get_partial_mnist_mirror_dataset,
                get_validation_mnist_mirror_dataset,
            ),
        }

        training_callback, validation_callback = mapping[self.name]
        training_dataset, input_shape, output_shape = training_callback()
        validation_dataset, _, _ = validation_callback()

        if self.batch_size == -1:
            batch_size = len(training_dataset)
        else:
            batch_size = self.batch_size

        train_loader = DataLoader(
            training_dataset,
            batch_size,
            shuffle=True,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size,
            shuffle=False,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )

        return train_loader, val_loader, input_shape, output_shape
