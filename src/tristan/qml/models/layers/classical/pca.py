import torch

from torch.nn import Module, Parameter

from qml.models.layers.utils import register
from qml.utils import single_to_int


class PCA(Module):
    def __init__(self, input_size: int, n_components: int):
        super().__init__()
        self.n_components = n_components
        self.mean = Parameter(torch.zeros((1, input_size)), requires_grad=False)
        self.components = Parameter(
            torch.zeros((input_size, n_components)), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: replace this by something more reliable
        if x.shape[0] > 100:
            self._fit(x)

        return self._transform(x)

    # Must be the full dataset
    def _fit(self, data: torch.Tensor):
        self.mean = Parameter(data=data.mean(dim=0, keepdim=True), requires_grad=False)
        centered_data = data - self.mean

        _, S, values = torch.pca_lowrank(centered_data, q=self.n_components)

        self.components = Parameter(
            data=values[:, : self.n_components], requires_grad=False
        )
        # self._display_eigenvalues(S)
        # self._display_pca(S, values, (28, 28))

    def _transform(self, data: torch.Tensor):
        # with torch.no_grad():
        centered_data = data - self.mean
        projected = torch.matmul(centered_data, self.components)
        # self._display_pca(projected, self.components.data, (28, 28))
        return projected

    def _display_eigenvalues(self, S):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(S, "-x")
        plt.xlabel("number of eigen vectors")
        plt.ylabel("(value of) eigen values")
        plt.grid(visible=True, which="major")
        plt.show()

    # "S" and "V" are the tensors returned by torch.pca_lowrank
    # reshape is an optional tuple to reshape the vector
    # See https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
    def _display_pca(self, S, V: torch.Tensor, reshape=None):
        import matplotlib.pyplot as plt
        import math

        S = S.squeeze()
        if len(S.shape) > 1:
            print(f"can only display PCA for one item, found {S.shape}")
            return
        n = math.ceil(math.sqrt(self.n_components))
        fig, axs = plt.subplots(nrows=n, ncols=n)
        for i in range(self.n_components):
            ax = axs[i // n][i % n]
            vector = V[:, i]
            if reshape is not None:
                vector = torch.reshape(vector, reshape)
            ax.imshow(vector)
            ax.set_title("Vector %d, eigen value=%.1f" % (i, S[i]))
        plt.show()


@register
class PCABuilder:
    TYPE = "pca"

    @classmethod
    def elem(cls, n_components: int) -> dict:
        return dict(type=cls.TYPE, n_components=n_components)

    @classmethod
    def make(cls, input_size, n_components, **kwargs) -> Module:
        return PCA(single_to_int(input_size), n_components)

    @classmethod
    def predict_size(cls, n_components, **kwargs) -> tuple:
        return (n_components,)
