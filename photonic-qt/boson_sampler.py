import perceval as pcvl
import torch
from math import comb
from typing import Iterable
from functools import lru_cache
from itertools import combinations

## Quantum-Train Inspired ##
class BosonSampler:
    def __init__(self, m: int, n: int, postselect: int = None, session=None):
        """
        A class that combines the Boson Sampler with quantum-train's idea by integrating
        parametrized quantum logic gates and a classical neural network.

        :param m: Number of modes in the photonic circuit.
        :param n: Number of photons input into the circuit.
        :param postselect: Minimum number of detected photons for an output state to be valid.
        :param session: Optional Perceval session for remote simulation.
        """
        self.m = m
        self.n = n
        assert n <= m, "Got more modes than photons, can only input 0 or 1 photon per mode"
        self.postselect = postselect or n
        assert self.postselect <= n, "Postselect cannot require more photons than are input."
        self.session = session

    @property
    def _nb_parameters_needed(self) -> int:
        """Number of parameters (theta, phi_i, phi_j) in the circuit."""
        # Each beam splitter has theta, and each connected mode has a phase shifter
        # Number of beam splitters in Clements decomposition: m(m-1)/2
        # Thus, total parameters: m(m-1)/2 * 3
        return (self.m * (self.m - 1)) // 2 * 3

    @property
    def nb_parameters(self) -> int:
        """Maximum number of values in the input tensor."""
        return self._nb_parameters_needed

    def create_circuit(self, parameters: Iterable[float] = None, qnn_layers = 1) -> pcvl.Circuit:
        """
        Creates a parametrized interferometer using Clements decomposition.

        :param parameters: Iterable of phase parameters.
        :return: A Perceval Circuit object.
        """
        if parameters is None:
            parameters = [torch.rand(1).item() for _ in range(self.nb_parameters * qnn_layers)]
        else:
            parameters = list(parameters)

        if len(parameters) < self.nb_parameters * qnn_layers:
            # Pad with zeros if not enough parameters
            parameters += [0.0] * (self.nb_parameters * qnn_layers - len(parameters))
        elif len(parameters) > self.nb_parameters * qnn_layers:
            raise ValueError(f"Too many parameters provided. Expected {self.nb_parameters}, got {len(parameters)}.")

        circuit = pcvl.Circuit(self.m)
        param_idx = 0

        num_layers = self.m - 1

        for layer in range(num_layers):
            # Determine starting mode based on layer parity for checkerboard pattern
            if layer % 2 == 0:
                mode_start = 0
            else:
                mode_start = 1

            for mode in range(mode_start, self.m - 1, 2):
                i = mode
                j = mode + 1

                # Add Beam Splitter with theta parameter
                bs_theta = parameters[param_idx]
                param_idx += 1
                bs = pcvl.BS(theta=bs_theta)
                circuit.add((i, j), bs)

                # Add Phase Shifters to modes i and j
                ps_i = parameters[param_idx]
                param_idx += 1
                circuit.add(i, pcvl.PS(ps_i))

                ps_j = parameters[param_idx]
                param_idx += 1
                circuit.add(j, pcvl.PS(ps_j))
                
        if qnn_layers > 1:
            for qnn_layer in range(qnn_layers-1):
                for mode in range(self.m):
                    circuit.add(mode, pcvl.PS(3.1415926)) ## non-linear layer
                    
                for layer in range(num_layers):
                    # Determine starting mode based on layer parity for checkerboard pattern
                    if layer % 2 == 0:
                        mode_start = 0
                    else:
                        mode_start = 1

                    for mode in range(mode_start, self.m - 1, 2):
                        i = mode
                        j = mode + 1

                        # Add Beam Splitter with theta parameter
                        bs_theta = parameters[param_idx]
                        param_idx += 1
                        bs = pcvl.BS(theta=bs_theta)
                        circuit.add((i, j), bs)

                        # Add Phase Shifters to modes i and j
                        ps_i = parameters[param_idx]
                        param_idx += 1
                        circuit.add(i, pcvl.PS(ps_i))

                        ps_j = parameters[param_idx]
                        param_idx += 1
                        circuit.add(j, pcvl.PS(ps_j))
                    
            
        return circuit

    def embed(self, t: torch.tensor, n_sample: int) -> torch.tensor:
        """
        Embeds the tensor t using its values as phases in a circuit and returns the output probability distribution.

        :param t: The tensor to be embedded, with values between 0 and 1.
        :param n_sample: The number of samples used to estimate the output probability distribution.
        :return: A 1D tensor representing the output probability distribution.
        """
        t = t.reshape(-1)
        if len(t) > self.nb_parameters:
            raise ValueError(f"Too many parameters (got {len(t)}, maximum {self.nb_parameters}).")

        # Complete the tensor to have the correct number of phases
        if len(t) < self.nb_parameters:
            z = torch.zeros(self.nb_parameters - len(t))
            t = torch.cat((t, z), dim=0)

        t = t * 2 * torch.pi  # Scale the values to the [0, 2π] interval for phases

        # Convert tensor to list of floats
        parameters = t.tolist()

        # Run the boson sampler with the parametrized circuit
        res = self.run(parameters, n_sample)

        # Get the output probabilities as a tensor
        probs = self.translate_results(res)

        return probs

    @property
    def embedding_size(self) -> int:
        """Size of the output probability distribution."""
        s = 0
        for k in range(self.postselect, self.n + 1):
            s += comb(self.m, k)
        return s

    def translate_results(self, res: dict) -> torch.tensor:
        """Transforms the results into a tensor of probabilities."""
        state_list = self.generate_state_list()
        t = torch.zeros(self.embedding_size)
        total_counts = sum(res.values())
        for i, state in enumerate(state_list):
            count = res.get(state, 0)
            t[i] = count / total_counts  # Normalize to get probabilities
        return t

    @lru_cache
    def generate_state_list(self) -> list:
        """Generates all possible output states."""
        res = []
        for k in range(self.postselect, self.n + 1):
            res += self._generate_state_list_k(k)
        return res

    def _generate_state_list_k(self, k) -> list:
        """Generates all binary states of size self.m with exactly k ones."""
        res = []
        for indices in combinations(range(self.m), k):
            state = [0] * self.m
            for idx in indices:
                state[idx] = 1
            res.append(pcvl.BasicState(state))
        return res

    def prepare_processor(self, processor, parameters: Iterable[float], qnn_layers: int) -> None:
        """Give the important info to the processor"""
        processor.set_circuit(self.create_circuit(parameters, qnn_layers))
        processor.min_detected_photons_filter(self.postselect)
        # processor.thresholded_output(True)
        
        # Evenly spaces the photons
        input_state = self.m * [0]
        places = torch.linspace(0, self.m - 1, self.n)
        for photon in places:
            input_state[int(photon)] = 1
        input_state = pcvl.BasicState(input_state)
        
        processor.with_input(input_state)

        
    def run(self, parameters: Iterable[float], samples: int, qnn_layers=1) -> dict:
        """Samples and return the raw results, using the parameters as circuit phases"""
        if self.session is not None:
            proc = self.session.build_remote_processor()
        else:
            # Local simulation
            proc = pcvl.Processor("SLOS", self.create_circuit(parameters, qnn_layers))

        self.prepare_processor(proc, parameters, qnn_layers)

        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=samples)
        res = sampler.probs(samples)
            
        return res["results"]
