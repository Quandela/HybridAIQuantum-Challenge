import perceval as pcvl
import torch
from math import comb
from typing import Iterable
from functools import lru_cache

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BosonSampler:
    
    def __init__(self, m: int, n: int, postselect: int = None, session: pcvl.ISession = None):
        """
        A class able to embed a tensor using a photonic circuit with thresholded outputs.
        
        :param m: The number of modes of the circuit.
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
        return self.m * (self.m - 1)
    
    @property
    def nb_parameters(self) -> int:
        return self._nb_parameters_needed - (self.m // 2)
    
    def create_circuit(self, parameters: Iterable[float] = None) -> pcvl.Circuit:
        if parameters is None:
            parameters = [p for i in range(self.m * (self.m - 1) // 2)
                          for p in [pcvl.P(f"phi_{2 * i}"), pcvl.P(f"phi_{2 * i + 1}")]]
        return pcvl.GenericInterferometer(
            self.m,
            lambda i: (pcvl.BS()
                       .add(0, pcvl.PS(parameters[2 * i]))
                       .add(0, pcvl.BS())
                       .add(0, pcvl.PS(parameters[2 * i + 1])))
        )
        
    def embed(self, t: torch.tensor, n_sample: int) -> torch.tensor:
        t = t.reshape(-1)
        if len(t) > self.nb_parameters:
            raise ValueError(f"Got too many parameters (got {len(t)}, maximum {self.nb_parameters})")
        
        z = torch.zeros(self._nb_parameters_needed - len(t))
        if len(z):
            t = torch.cat((t, z), 0)
            
        t = t * 2 * torch.pi
        mode_probs = self.run(t, n_sample)
        return mode_probs
        
    def run(self, parameters: Iterable[float], samples: int) -> torch.tensor:
        if self.session is not None:
            proc = self.session.build_remote_processor()
        else:
            proc = pcvl.Processor("CliffordClifford2017", self.m)

        self.prepare_processor(proc, parameters)
        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=samples)
        results = sampler.probs(samples)
        state_distribution = results["results"]

        # Calculate the photon number expectation value
        
        states = list(state_distribution.keys())
        probs = torch.tensor(list(state_distribution.values()), device=device, dtype=torch.float32)
        
        state_matrix = torch.stack([
            torch.tensor(list(state), device=device, dtype=torch.float32) for state in states
        ])
        
        mode_probs = state_matrix.t().mv(probs)
        return mode_probs
        
    @property
    def embedding_size(self) -> int:
        s = 0
        for k in range(self.postselect, self.n + 1):
            s += comb(self.m, k)
        return s
        
    def translate_results(self, res: pcvl.BSDistribution) -> torch.tensor:
        state_list = self.generate_state_list()
        t = torch.zeros(self.embedding_size)
        for i, state in enumerate(state_list):
            t[i] = res[state]
        return t
        
    @lru_cache
    def generate_state_list(self) -> list:
        res = []
        for k in range(self.postselect, self.n + 1):
            res += self._generate_state_list_k(k)
        return res
    
    def _generate_state_list_k(self, k) -> list:
        return list(map(pcvl.BasicState, pcvl.utils.qmath.distinct_permutations(k * [1] + (self.m - k) * [0])))
        
    def prepare_processor(self, processor, parameters: Iterable[float]) -> None:
        processor.set_circuit(self.create_circuit(parameters))
        processor.min_detected_photons_filter(self.postselect)
        processor.thresholded_output(True)
        
        input_state = self.m * [0]
        places = torch.linspace(0, self.m - 1, self.n)
        for photon in places:
            input_state[int(photon)] = 1
        input_state = pcvl.BasicState(input_state)
        processor.with_input(input_state)
