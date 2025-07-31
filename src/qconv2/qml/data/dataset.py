import re
import torch

from torch.utils.data import Dataset


_MNIST_IMAGE_SHAPE = (1, 28, 28)
_MNIST_CLASSES_SHAPE = (10,)


class LocalMnistDataset(Dataset):
    def __init__(self, path: str = ".", transform=None):
        from pandas import read_csv

        self._transform = transform
        self._df = read_csv(path)

    def __len__(self):
        l = len(self._df["image"])
        return l

    def __getitem__(self, idx):
        img = self._df["image"].iloc[idx]
        y = self._df["label"].iloc[idx]
        # string to list
        img_list = re.split(r",", img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        x = torch.unflatten(torch.tensor(img_float), 0, _MNIST_IMAGE_SHAPE)

        if self._transform is not None:
            x = self._transform(x)

        return x, y


class LocalMirrorMnistDataset(Dataset):
    def __init__(self, path: str = ".", transform=None):
        from pandas import read_csv

        self._transform = transform
        self._df = read_csv(path)

    def __len__(self):
        return len(self._df["image"])

    def __getitem__(self, idx):
        img = self._df["image"].iloc[idx]
        # string to list
        img_list = re.split(r",", img)
        # remove '[' and ']'
        img_list[0] = img_list[0][1:]
        img_list[-1] = img_list[-1][:-1]
        # convert to float
        img_float = [float(el) for el in img_list]
        # convert to image
        x = torch.unflatten(torch.tensor(img_float), 0, _MNIST_IMAGE_SHAPE)

        if self._transform is not None:
            x = self._transform(x)

        return x, x


def get_partial_mnist_mirror_dataset():
    training_dataset = LocalMirrorMnistDataset(
        path="./resources/mnist_partial/train.csv"
    )

    return training_dataset, _MNIST_IMAGE_SHAPE, _MNIST_IMAGE_SHAPE


def get_partial_mnist_classification_dataset():
    training_dataset = LocalMnistDataset(path="./resources/mnist_partial/train.csv")

    return training_dataset, _MNIST_IMAGE_SHAPE, _MNIST_CLASSES_SHAPE


def get_full_mnist_classification_dataset():
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    training_dataset = MNIST(
        root="./resources/mnist_full", download=True, transform=ToTensor()
    )

    return training_dataset, _MNIST_IMAGE_SHAPE, _MNIST_CLASSES_SHAPE


def get_validation_mnist_classification_dataset():
    validation_dataset = LocalMnistDataset(path="./resources/mnist_partial/val.csv")

    return validation_dataset, _MNIST_IMAGE_SHAPE, _MNIST_CLASSES_SHAPE


def get_validation_mnist_mirror_dataset():
    validation_dataset = LocalMirrorMnistDataset(
        path="./resources/mnist_partial/val.csv"
    )

    return validation_dataset, _MNIST_IMAGE_SHAPE, _MNIST_IMAGE_SHAPE
