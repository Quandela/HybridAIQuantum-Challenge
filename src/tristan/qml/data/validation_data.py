from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch import Generator

from qml.data.dataset import (
    get_validation_mnist_classification_dataset,
    get_validation_mnist_mirror_dataset,
)


@dataclass
class ValidationData:
    batch_size: int
    name: str
    device: str = None

    def build_loaders(self):
        mapping = {
            "mnist_classification": get_validation_mnist_classification_dataset,
            "mnist_mirror": get_validation_mnist_mirror_dataset,
        }

        validation_dataset, input_shape, output_shape = mapping[self.name]()

        val_loader = DataLoader(
            validation_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=6,
            persistent_workers=True,
            generator=Generator(device=self.device),
        )

        return val_loader, input_shape, output_shape
