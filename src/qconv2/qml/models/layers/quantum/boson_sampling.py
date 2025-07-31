import perceval as pcvl
import torch

from torch.nn import Module
from math import comb

from typing import Iterable, Optional

from qml.models.layers.quantum.converters import build_converter
from qml.models.layers.quantum.runners.utils import build_session
from qml.models.layers.utils import register


class BosonSampling(Module):
    def __init__(
        self,
        modes: int,
        photons: int,
        samples: int,
        select: Optional[int] = None,
        session_name: Optional[str] = None,
    ):
        super().__init__()
        self._modes = modes
        self._photons = photons
        self._samples = samples

        assert (
            photons <= modes
        ), "Got more photons than modes, can only input 0 or 1 photon per mode"
        self._min_photons = select or photons

        assert (
            self._min_photons <= photons
        ), "Cannot postselect with more photons than the input number of photons"

        self._session = (
            build_session(session_name, "sim:sampling:p100") if session_name else None
        )

        self._conv = build_converter("default")

    @property
    def _nb_parameters_needed(self) -> int:
        """Returns the number of phase shifters in the circuit. Only used internally"""
        return self._modes * (self._modes - 1)

    @property
    def nb_parameters(self) -> int:
        """Returns the maximum number of values in the input tensor.
        This corresponds to the number of phase shifters that can affect the output probabilities in the circuit
        """
        return self._nb_parameters_needed - (
            self._modes // 2
        )  # Doesn't count the last layer of PS as it doesn't change anything

    def _create_circuit(self, parameters: Iterable[float] = None) -> pcvl.Circuit:
        """Creates a generic interferometer using a list of phases of size self._nb_parameters_needed.
        If no list is provided, the circuit is built with perceval parameters"""
        if parameters is None:
            parameters = [
                p
                for i in range(self._modes * (self._modes - 1) // 2)
                for p in [pcvl.P(f"phi_{2 * i}"), pcvl.P(f"phi_{2 * i + 1}")]
            ]
        return pcvl.GenericInterferometer(
            self._modes,
            lambda i: (
                pcvl.BS()
                .add(0, pcvl.PS(parameters[2 * i]))
                .add(0, pcvl.BS())
                .add(0, pcvl.PS(parameters[2 * i + 1]))
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds the tensor x using its values as phases in a circuit, and returns the output probability distribution
        """

        assert (
            x.shape[0] == 1
        ), f"Boso sampling `forward` only handles batch size of 1, got {x.shape[0]}"

        t = x.reshape(-1)  # We need to see t as a list of values
        if len(t) > self.nb_parameters:
            raise ValueError(
                f"Got too many parameters (got {len(t)}, maximum {self.nb_parameters})"
            )

        # We need to complete the tensor to have the good number of phases
        z = torch.zeros(self._nb_parameters_needed - len(t))
        if len(z):
            t = torch.cat((t, z), 0)

        # Phases are 2 pi periodic --> we get better expressivity by multiplying the values by 2 pi
        t = t * 2 * torch.pi

        # This is a dict with states as keys and probabilities as values
        res = self._run(t, self._samples)

        return self._conv.to_tensor(res, self._photons, self._min_photons)

    def _prepare_processor(self, processor, parameters: Iterable[float]) -> None:
        """Give the important info to the processor"""
        processor.set_circuit(self._create_circuit(parameters))
        processor.min_detected_photons_filter(self._min_photons)
        processor.thresholded_output(True)

        # Evenly spaces the photons
        input_state = self._modes * [0]
        places = torch.linspace(0, self._modes - 1, self._photons)
        for photon in places:
            input_state[int(photon)] = 1
        input_state = pcvl.BasicState(input_state)

        processor.with_input(input_state)

    def _run(self, parameters: Iterable[float], samples: int) -> pcvl.BSDistribution:
        """Samples and return the raw results, using the parameters as circuit phases"""
        if self._session is not None:
            proc = self._session.build_remote_processor()

        else:
            # Local simulation
            proc = pcvl.Processor("SLOS", self._modes)

        self._prepare_processor(proc, parameters)

        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=samples)
        res = sampler.probs(samples)

        return res["results"]


@register
class BosonSamplingBuilder:
    TYPE = "boson_sampling"

    @classmethod
    def elem(
        cls,
        modes: int,
        photons: int,
        samples: int = 1,
        select: Optional[int] = None,
        session: Optional[str] = None,
    ) -> dict:
        return dict(
            type=cls.TYPE,
            modes=modes,
            samples=samples,
            photons=photons,
            select=select,
            session_name=session,
        )

    @classmethod
    def make(cls, modes, photons, samples, select, session_name, **kwargs) -> Module:
        return BosonSampling(modes, photons, samples, select, session_name)

    @classmethod
    def predict_size(cls, modes, photons, select, **kwargs) -> tuple:
        s = 0

        select = select or photons

        for k in range(select, photons + 1):
            s += comb(modes, k)

        return (s,)
