# for the Boson Sampler
import perceval as pcvl

import torch
from math import comb

from typing import Iterable

from functools import lru_cache

class BosonSampler:
    
    def __init__(self, m: int, n: int, postselect: int = None, session : pcvl.ISession = None):
        """
        A class able to embed a tensor using a photonic circuit witèh thresholded outputs.
        
        :param m: The number of modes of the circuit. Larger values allow more values in the embedded tensor.
        :param n: The number of photons to input in the circuit.
        :param postselect: The minimum number of detected photons to count an output state as valid. Defaults to n.
        :param session: An optional scaleway session. If provided, simulations will be launched remotely, else they will run locally.
        """
        self.m = m
        self.n = n
        assert n <= m, "Got more modes than photons, can only input 0 or 1 photon per mode"
        self.postselect = postselect or n
        assert self.postselect <= n, "Cannot postselect with more photons than the input number of photons"
        self.session = session

    @property
    def _nb_parameters_needed(self) -> int:
        """Returns the number of phase shifters in the circuit. Only used internally"""
        return self.m * (self.m - 1)
    
    @property
    def nb_parameters(self) -> int:
        """Returns the maximum number of values in the input tensor.
          This corresponds to the number of phase shifters that can affect the output probabilities in the circuit"""
        return self._nb_parameters_needed - (self.m // 2)  # Doesn't count the last layer of PS as it doesn't change anything
    
    def create_circuit(self, parameters: Iterable[float] = None) -> pcvl.Circuit:
        """Creates a generic interferometer using a list of phases of size self._nb_parameters_needed.
        If no list is provided, the circuit is built with perceval parameters"""
        if parameters is None:
            parameters = [p for i in range(self.m * (self.m - 1) // 2)
                            for p in [pcvl.P(f"phi_{2 * i}"), pcvl.P(f"phi_{2 * i + 1}")]]
        return pcvl.GenericInterferometer(self.m, lambda i: (pcvl.BS()
                                                             .add(0, pcvl.PS(parameters[2 * i]))
                                                             .add(0, pcvl.BS())
                                                             .add(0, pcvl.PS(parameters[2 * i + 1]))
                                                             )
                                          )
        
    def embed(self, t: torch.tensor, n_sample: int) -> torch.tensor:
        """
        Embeds the tensor t using its values as phases in a circuit, and returns the output probability distribution

        :param t: The tensor to be embedded, with values between 0 and 1
        :param n_sample: The number of samples used to estimate the output probability distribution. Not used if running on a simulator
        :return: A 1D tensor of size self.embedding_size representing the output probability distribution, estimated using n_sample"""

        t = t.reshape(-1)  # We need to see t as a list of values
        if len(t) > self.nb_parameters:
            raise ValueError(f"Got too many parameters (got {len(t)}, maximum {self.nb_parameters})")
        
        # We need to complete the tensor to have the good number of phases
        z = torch.zeros(self._nb_parameters_needed - len(t))
        if len(z):
            t = torch.cat((t, z), 0)
            
        t = t * 2 * torch.pi  # Phases are 2 pi periodic --> we get better expressivity by multiplying the values by 2 pi
        
        res = self.run(t, n_sample)  # This is a dict with states as keys and probabilities as values
        
        return self.translate_results(res)  # We need to transform this dict into a tensor
        
    @property
    def embedding_size(self) -> int:
        """Size of the returned tensor. This is the number of possible output states"""
        # For thresholded output, this is the number of binary numbers having at least self.postselect 1s
        s = 0
        for k in range(self.postselect, self.n + 1):
            s += comb(self.m, k)
        return s
        
    def translate_results(self, res: pcvl.BSDistribution) -> torch.tensor:
        """Transforms the perceval results into a list of probabilities, where each output is always represented at the same position"""
        
        # First, we generate a list of all possible output states
        state_list = self.generate_state_list()
        
        # Then we take the probabilities from the BSD in the order of the list
        t = torch.zeros(self.embedding_size)
        for i, state in enumerate(state_list):
            t[i] = res[state]
            
        return t
        
    @lru_cache  # Always the same, no need to compute it each time
    def generate_state_list(self) -> list:
        """Generate a list of all possible output states"""
        res = []
        for k in range(self.postselect, self.n + 1):
            res += self._generate_state_list_k(k)
        
        return res
    
    def _generate_state_list_k(self, k) -> list:
        """Generate all binary states of size self.m having exactly *k* 1s"""
        return list(map(pcvl.BasicState, pcvl.utils.qmath.distinct_permutations(k * [1] + (self.m - k) * [0])))
        
        
    def prepare_processor(self, processor, parameters: Iterable[float]) -> None:
        """Give the important info to the processor"""
        processor.set_circuit(self.create_circuit(parameters))
        processor.min_detected_photons_filter(self.postselect)
        processor.thresholded_output(True)
        
        # Evenly spaces the photons
        input_state = self.m * [0]
        places = torch.linspace(0, self.m - 1, self.n)
        for photon in places:
            input_state[int(photon)] = 1
        input_state = pcvl.BasicState(input_state)
        
        processor.with_input(input_state)
        
    def run(self, parameters: Iterable[float], samples: int) -> pcvl.BSDistribution:
        """Samples and return the raw results, using the parameters as circuit phases"""
        if self.session is not None:
            proc = self.session.build_remote_processor()

        else:
            # Local simulation
            proc = pcvl.Processor("SLOS", self.m)

        self.prepare_processor(proc, parameters)

        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=samples)
        res = sampler.probs(samples)
            
        return res["results"]
