import perceval as pcvl
import abc

from typing import Optional

from qml.models.layers.quantum.runners.register import register, runner_factory
from qml.models.layers.quantum.runners.utils import build_session


def build_runner(name: str, **kwargs):
    return runner_factory()[name].make(**kwargs)


class Runner(abc.ABC):
    @abc.abstractmethod
    def run(
        circuit: pcvl.Circuit,
        input_state: str,
        samples: int,
        min_photon: int,
        post_select: str,
        thresholded_output: bool,
    ) -> pcvl.BSDistribution:
        pass


class AnubisRunner(Runner):
    def __init__(
        self,
        session_name: Optional[str] = None,
        platform_name: Optional[str] = None,
    ):
        super().__init__()

        self._session = None

        if session_name:
            platform_name = "sim:sampling:p100" if not platform_name else platform_name
            self._session = build_session(session_name, platform_name)

    def run(
        self,
        circuit: pcvl.Circuit,
        input_state: str,
        samples: int,
        min_photon: int,
        post_select: str,
        thresholded_output: bool,
    ) -> pcvl.BSDistribution:
        if self._session is not None:
            processor = self._session.build_remote_processor()
        else:
            processor = pcvl.Processor("SLOS", circuit.m)

        if thresholded_output:
            for i in range(circuit.m):
                processor.add(i, pcvl.Detector.threshold())
        else:
            for i in range(circuit.m):
                processor.add(i, pcvl.Detector.pnr())

        if isinstance(input_state, str):
            input_state = pcvl.BasicState(input_state)

        processor.set_circuit(circuit)
        processor.min_detected_photons_filter(min_photon)

        if post_select:
            processor.set_postselection(pcvl.PostSelect(post_select))

        processor.with_input(input_state)

        sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=samples)
        job_results = sampler.probs(samples)

        return job_results["results"]


@register
class AnubisRunnerBuilder:
    TYPE = "anubis"

    @classmethod
    def make(cls, session_name, platform_name, **kwargs) -> AnubisRunner:
        return AnubisRunner(session_name, platform_name)
