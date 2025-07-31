import torch

from torch.nn import Module, Parameter
from torch.autograd import Function

from typing import Optional, Tuple

from qml.models.layers.utils import register
from qml.models.layers.quantum.ansatz import (
    build_ansatz,
    predict_size_ansatz,
    build_weights,
)
from qml.models.layers.quantum.feature_maps import (
    build_feature_map,
    predict_size_feature_map,
    FeatureMapParams,
)
from qml.models.layers.quantum.runners import build_runner
from qml.models.layers.quantum.post_selecters import build_post_select
from qml.models.layers.quantum.gradients import build_gradient_method
from qml.models.layers.quantum.converters import build_converter, predict_size_converter


class _ComputeFwAndBw(Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        weights: torch.Tensor,
        forward_cb: callable,
        backward_cb: callable,
    ):
        return forward_cb(x, weights)

    @staticmethod
    def setup_context(ctx, inputs: torch.Tensor, y: torch.Tensor):
        (
            x,
            weights,
            forward_cb,
            backward_cb,
        ) = inputs

        ctx.save_for_backward(x, weights)
        ctx.forward_cb = forward_cb  # /!\ Warning: this attribute name is used _as_it_ in the backward_cb call
        ctx.backward_cb = backward_cb

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.backward_cb(ctx, grad_output)


class PhotonicCircuit(Module):
    def __init__(
        self,
        samples: int,
        fmap_name: str,
        runner_name: str,
        ansatz_name: Optional[str] = None,
        session_name: Optional[str] = None,
        platform_name: Optional[str] = None,
        input_size: Optional[Tuple] = None,
        post_select_name: Optional[str] = None,
        converter_name: Optional[str] = None,
        gradient_method_name: Optional[str] = None,
        thresholded_output: bool = False,
        iterations: int = 1,
    ):
        super().__init__()

        self._fmap_name = fmap_name
        self._ansatz_name = ansatz_name
        self._samples = samples
        self._weights = None
        self._backward = None
        self._iterations = iterations
        self._post_select_name = post_select_name
        self._converter_name = converter_name
        self._thresholded_output = thresholded_output

        assert iterations > 0

        if self._ansatz_name:
            assert (
                gradient_method_name
            ), "Cannot have ansatz without None gradient method"

            self._weights = Parameter(
                build_weights(ansatz_name, input_size, iterations=iterations)
            )

            assert (
                len(self._weights.shape) == 2
            ), f"Built weights shape must be 2-dim, got {len(self._weights.shape)}"

            self._backward = build_gradient_method(gradient_method_name)

            assert self._backward

        self._runner = build_runner(
            name=runner_name, session_name=session_name, platform_name=platform_name
        )

        assert self._runner

        self._converter = build_converter(
            name=converter_name, thresholded_output=thresholded_output
        )

        assert self._converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._pcvl_forward(x)

    def _pcvl_forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[0] == 1
        ), f"Photonic circuit `forward` only handles batch size of 1, got {x.shape[0]}"

        def _forward(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            post_select = None

            circuit, params = self._build_iterative_circuit(x, weights)

            if self._post_select_name:
                post_select = build_post_select(self._post_select_name, circuit.m)

            output_distribution = self._runner.run(
                circuit,
                params.input_state,
                self._samples,
                params.min_detect_photon,
                post_select,
                self._thresholded_output,
            )

            output_tensor = self._converter.to_tensor(
                output_distribution, params.input_state.n, params.min_detect_photon
            )

            return output_tensor

        # As ansatz are parametric quantum circuit, we apply a function to manage backpropagation
        if self._ansatz_name:
            return _ComputeFwAndBw.apply(x, self._weights, _forward, self._backward)

        return _forward(x, self._weights)

    def _build_iterative_circuit(self, x: torch.Tensor, weights: torch.Tensor):
        circuit = None

        for i in range(self._iterations):
            p = self._build_circuit(x, weights[i] if weights is not None else None)

            if not circuit:
                params = p
                circuit = p.circuit
            else:
                circuit.add(0, p.circuit)

        return circuit, params

    def _build_circuit(
        self, x: torch.Tensor, weights: torch.Tensor
    ) -> FeatureMapParams:
        fmap_params = build_feature_map(self._fmap_name, x)

        if self._ansatz_name:
            ansatz = build_ansatz(self._ansatz_name, x, weights)
            fmap_params.circuit.add(0, ansatz)

        return fmap_params


@register
class PhotonicCircuitBuilder:
    TYPE = "photonic_circuit"

    @classmethod
    def elem(
        cls,
        fmap: str,
        runner: str = "default",
        converter: str = "default",
        gradient_method: str = "spsa",
        thresholded_output: bool = False,
        ansatz: Optional[str] = None,
        samples: int = 1,
        session: Optional[str] = None,
        platform: Optional[str] = None,
        post_select: Optional[str] = None,
        iterations: int = 1,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            ansatz_name=ansatz,
            fmap_name=fmap,
            session_name=session,
            platform_name=platform,
            samples=samples,
            runner_name=runner,
            converter_name=converter,
            iterations=iterations,
            post_select_name=post_select,
            gradient_method_name=gradient_method,
            thresholded_output=thresholded_output,
        )

    @classmethod
    def make(
        cls,
        fmap_name,
        ansatz_name,
        samples,
        runner_name,
        converter_name,
        session_name,
        platform_name,
        input_size,
        post_select_name,
        gradient_method_name,
        thresholded_output,
        iterations,
        **kwargs,
    ) -> Module:
        return PhotonicCircuit(
            samples=samples,
            fmap_name=fmap_name,
            ansatz_name=ansatz_name,
            session_name=session_name,
            runner_name=runner_name,
            input_size=input_size,
            platform_name=platform_name,
            post_select_name=post_select_name,
            converter_name=converter_name,
            iterations=iterations,
            gradient_method_name=gradient_method_name,
            thresholded_output=thresholded_output,
        )

    @classmethod
    def predict_size(
        cls,
        input_size,
        fmap_name,
        ansatz_name,
        converter_name,
        thresholded_output,
        **kwargs,
    ) -> tuple:
        # m, n, min_n
        output_size = predict_size_feature_map(fmap_name, input_size)

        if ansatz_name:
            # m, n, min_n
            output_size = predict_size_ansatz(ansatz_name, output_size)

        # tensor shape
        output_size = predict_size_converter(
            converter_name, output_size, thresholded_output=thresholded_output
        )

        return output_size
