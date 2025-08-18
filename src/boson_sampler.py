# src/boson_sampler.py
import perceval as pcvl
from perceval.providers import scaleway  # import Scaleway Session class if needed

import torch
import torch.nn as nn
import os
import hashlib
import pickle
import numpy as np
from scipy.signal import convolve2d

from abc import ABC, abstractmethod


def merge_circuits(c1, c2):
    """
    Merge two circuits by creating a new composite circuit.
    This helper creates a new circuit with the same number of modes as c1,
    and adds c1 and then c2 using the add() method with merge=True.
    """
    composite = pcvl.Circuit(c1.m)
    # Add the first circuit on port 0 and merge it into the composite circuit.
    composite.add(0, c1, merge=True)
    # Add the second circuit similarly.
    composite.add(0, c2, merge=True)
    return composite

def compose_circuits(circuit_list):
    """
    Compose a list of circuits into a single circuit using merge_circuits.
    """
    composite = circuit_list[0]
    for circ in circuit_list[1:]:
        composite = merge_circuits(composite, circ)
    return composite




class InterferometerBuilder(ABC):
    @abstractmethod
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        """Return a Perceval circuit for m modes given a list of phase parameters."""
        pass

def safe_param(i, params):
    # If i is within bounds, return the parameter; otherwise, default to 0.
    return params[i] if i < len(params) else 0

class TriangularInterferometerBuilder(InterferometerBuilder):
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        if parameters is None:
            # Use symbolic parameters: for a triangular mesh you need m*(m-1) parameters.
            parameters = [pcvl.P(f"phi_{i}") for i in range(m * (m - 1))]
        # Build the circuit using the local variable m and the correct variable name (parameters)
        return pcvl.GenericInterferometer(m, lambda idx: (pcvl.BS()
                                                             .add(0, pcvl.PS(safe_param(2 * idx, parameters)))
                                                             .add(0, pcvl.BS())
                                                             .add(0, pcvl.PS(safe_param(2 * idx+1, parameters)))
                                                             )
                                          ,shape=pcvl.InterferometerShape.TRIANGLE)


class RectangularInterferometerBuilder(InterferometerBuilder):
    def create_circuit(self, m: int, parameters: list[float] = None) -> pcvl.Circuit:
        if parameters is None:
            parameters = [pcvl.P(f"phi_rect_{i}") for i in range(m * (m - 1))]
        # For demonstration, we build two layers sequentially.
        # (Note: In practice, you’d implement the proper rectangular mesh.)
        
        return pcvl.GenericInterferometer(m, lambda idx: (pcvl.BS()
                                                             .add(0, pcvl.PS(safe_param(2 * idx, parameters)))
                                                             .add(0, pcvl.BS())
                                                             .add(0, pcvl.PS(safe_param(2 * idx+1, parameters)))
                                                             )
                                          ,shape=pcvl.InterferometerShape.RECTANGLE)



class PdfInterferometerBuilder:
    def create_circuit(self, m: int, parameters=None) -> pcvl.Circuit:
        """
        Create a circuit as described in Quantum_Circuits.pdf.
        Expects 'parameters' to be either:
          - A dict with keys 'theta', 'alpha', and 'beta'
            where 'theta' is a list of three phase values,
                  'alpha' is a list of two values for the first BS block, and
                  'beta' is a list of two values for the second BS block,
          - Or a list of at least 7 numbers, which will be interpreted as
            theta[0:3], alpha[0:2], beta[0:2].
        """
        if parameters is None:
            parameters = {
                'theta': [pcvl.P("theta1"), pcvl.P("theta2"), pcvl.P("theta3")],
                'alpha': [0.1, 0.2],
                'beta': [0.3, 0.4]
            }
        elif not isinstance(parameters, dict):
            # Assume parameters is a list; require at least 7 values
            if len(parameters) < 7:
                raise ValueError("PDFInterferometerBuilder requires at least 7 parameters when provided as a list.")
            parameters = {
                'theta': parameters[:3],
                'alpha': parameters[3:5],
                'beta': parameters[5:7]
            }
        
        def U_block(m, theta):
            sub = pcvl.Circuit(m)
            # Apply a phase shift on each mode with the same theta.
            for mode in range(m):
                sub.add(mode, pcvl.PS(theta), merge=True)
            return sub
        
        def BS_block(m, bs_params):
            sub = pcvl.Circuit(m)
            # For demonstration, add a beam splitter between modes 0 and 1.
            sub.add((0, 1), pcvl.BS(), merge=True)
            return sub
        
        circuit = pcvl.Circuit(m)
        circuit = circuit.add(0, U_block(m, parameters['theta'][0]), merge=True)
        circuit = circuit.add(0, BS_block(m, parameters['alpha']), merge=True)
        circuit = circuit.add(0, U_block(m, parameters['theta'][1]), merge=True)
        circuit = circuit.add(0, BS_block(m, parameters['beta']), merge=True)
        circuit = circuit.add(0, U_block(m, parameters['theta'][2]), merge=True)
        return circuit


class BaseInterferometerBuilder:
    ###Is similar to rectangular but with a "hickup".
    def create_circuit(self, m: int, parameters: list = None) -> pcvl.Circuit:
        """
        Create an alternative interferometer circuit using a simplified rectangular layout.
        If no parameters are provided, generate a list of symbolic parameters of length m*(m-1).
        The lambda uses these parameters in pairs.
        """
        # Compute the expected total number of phases (should be m*(m-1))
        total_needed = m * (m - 1)
        if parameters is None:
            parameters = [pcvl.P(f"phi_{i}") for i in range(total_needed)]
        
        def base_lambda(i):
            # Use the actual length of parameters to avoid index errors.
            # Each pair of parameters is used for one layer.
            num_pairs = len(parameters) // 2
            j = i % num_pairs
            return (pcvl.BS()
                    .add(0, pcvl.PS(parameters[2 * j]), merge=True)
                    .add(0, pcvl.BS(), merge=True)
                    .add(0, pcvl.PS(parameters[2 * j + 1]), merge=True))
        
        circuit = pcvl.GenericInterferometer(m, base_lambda)
        return pcvl.GenericInterferometer(self.m, lambda idx: (pcvl.BS()
                                                             .add(0, pcvl.PS(safe_param(2 * idx, parameters)))
                                                             .add(0, pcvl.BS())
                                                             .add(0, pcvl.PS(safe_param(2 * idx+1, parameters)))
                                                             )
                                          )



class ConvolutionalInterferometerBuilder:
    def __init__(self, filterA, filterB, image_size=(28, 28)):
        """
        Initialize with filterA and filterB (numpy arrays) and the expected image size.
        For example, use filterA and filterB of shape (6,5) to yield 23x24=552 outputs.
        """
        self.filterA = filterA
        self.filterB = filterB
        self.image_size = image_size

    def create_circuit(self, m: int, parameters: list = None) -> pcvl.Circuit:
        """
        Constructs a full circuit that interleaves trainable U-blocks with convolutional encoding blocks.
        The conv encoding blocks use parameters for α and β.
        If 'parameters' is None, symbolic parameters are used for the conv blocks.
        Otherwise, 'parameters' is assumed to be a list containing [alpha_flat, beta_flat] concatenated.
        """
        expected_length = m * (m - 1)  # For a triangular interferometer
        
        # Expected output sizes from valid convolution:
        filterA_shape = self.filterA.shape  # e.g., (6,5)
        filterB_shape = self.filterB.shape  # e.g., (6,5)
        alpha_rows = self.image_size[0] - filterA_shape[0] + 1
        alpha_cols = self.image_size[1] - filterA_shape[1] + 1
        beta_rows  = self.image_size[0] - filterB_shape[0] + 1
        beta_cols  = self.image_size[1] - filterB_shape[1] + 1
        alpha_out_len = alpha_rows * alpha_cols  # e.g., 23*24 = 552
        beta_out_len  = beta_rows * beta_cols

        if parameters is None:
            # Use symbolic parameters for the conv blocks.
            alpha_params = [pcvl.P(f"alpha_{i}") for i in range(alpha_out_len)]
            beta_params  = [pcvl.P(f"beta_{j}") for j in range(beta_out_len)]
        else:
            # Split the provided parameter list into alpha and beta portions.
            alpha_params = parameters[:alpha_out_len]
            beta_params  = parameters[alpha_out_len:alpha_out_len+beta_out_len]

        # Truncate the conv parameters to the expected length.
        if len(alpha_params) > expected_length:
            alpha_params = alpha_params[:expected_length]
        if len(beta_params) > expected_length:
            beta_params = beta_params[:expected_length]
        # Optionally, pad if fewer (not likely with full convolution)
        if len(alpha_params) < expected_length:
            alpha_params.extend([0.0]*(expected_length - len(alpha_params)))
        if len(beta_params) < expected_length:
            beta_params.extend([0.0]*(expected_length - len(beta_params)))
        
        # Build sub-circuits:
        U1 = pcvl.GenericInterferometer(m, lambda idx: pcvl.BS() // pcvl.PS(pcvl.P(f"theta1_{idx}")), shape=pcvl.InterferometerShape.TRIANGLE)
        U2 = pcvl.GenericInterferometer(m, lambda idx: pcvl.BS() // pcvl.PS(pcvl.P(f"theta2_{idx}")), shape=pcvl.InterferometerShape.TRIANGLE)
        U3 = pcvl.GenericInterferometer(m, lambda idx: pcvl.BS() // pcvl.PS(pcvl.P(f"theta3_{idx}")), shape=pcvl.InterferometerShape.TRIANGLE)
        Conv1 = pcvl.GenericInterferometer(m, lambda idx: pcvl.BS() // pcvl.PS(alpha_params[idx]), shape=pcvl.InterferometerShape.TRIANGLE)
        Conv2 = pcvl.GenericInterferometer(m, lambda idx: pcvl.BS() // pcvl.PS(beta_params[idx]), shape=pcvl.InterferometerShape.TRIANGLE)
        
        # Compose the full circuit by merging the sub-circuits.
        full_circuit = compose_circuits([U1, Conv1, U2, Conv2, U3])
        return full_circuit



    def compute_conv_parameters(self, image_tensor: np.ndarray) -> list:
        """
        Given an input image (as a numpy array), compute the convolution outputs for filterA and filterB,
        normalize them to the [0, 2π] range, and return the concatenated phase parameter list.
        If the input is 1D, reshape it to 2D using a factorization of its length.
        """
        if image_tensor.ndim != 2:
            if image_tensor.ndim == 1:
                # Import the downsample shape helper from utils
                from utils import compute_downsample_shape
                h, w = compute_downsample_shape(image_tensor.size)
                image_tensor = image_tensor.reshape((h, w))
            else:
                raise ValueError("Input image_tensor must be 2D after squeezing.")
        
        # Compute valid convolutions.
        alpha = convolve2d(image_tensor, self.filterA, mode="valid")
        beta  = convolve2d(image_tensor, self.filterB, mode="valid")
        # Flatten and normalize: here we use modulo 1 and scale to 2π (adjust normalization as needed)
        alpha_flat = (alpha.flatten() % 1.0) * 2 * np.pi
        beta_flat  = (beta.flatten() % 1.0) * 2 * np.pi
        param_vec = np.concatenate([alpha_flat, beta_flat])
        return param_vec.tolist()





class BosonSampler:
    def __init__(self, m: int, n: int, postselect: int = None, backend: str = "SLOS", session=None, builder: InterferometerBuilder = None, cache_enabled=False, cache_directory="results/cache"):
        """
        Photonic boson sampler for embedding data.
        :param m: number of modes
        :param n: number of photons (n <= m)
        :param postselect: minimum detected photons for a valid output (defaults to n if None)
        :param backend: local simulation backend name (ignored if session is provided)
        :param session: optional Perceval session for remote execution (Scaleway)
        :param builder: An instance of InterferometerBuilder to build the circuit.
        """
        assert n <= m, "Photons n must be <= modes m"
        self.m = m
        self.n = n
        self.postselect = postselect if postselect is not None else n
        assert self.postselect <= n, "Postselect cannot exceed number of photons"
        self.backend = backend
        self.session = session  # If not None, use remote session for simulation
        # If no builder is given, use the triangular mesh as default.
        if builder is None:
            from boson_sampler import TriangularInterferometerBuilder  # ensure proper import
            self.builder = TriangularInterferometerBuilder()
        else:
            self.builder = builder

        # Caching settings
        self.cache_enabled = cache_enabled
        self.cache_directory = cache_directory
        if self.cache_enabled and not os.path.exists(self.cache_directory):
            os.makedirs(self.cache_directory)


    def _compute_hash(self, data_tensor):
        # Compute a unique hash based on the tensor contents.
        # Ensure the tensor is on CPU and convert to a numpy byte string.
        return hashlib.md5(data_tensor.cpu().numpy().tobytes()).hexdigest()


    @property
    def nb_parameters(self) -> int:
        """Number of phase parameters available (used for embedding)."""
        return self.m * (self.m - 1) - (self.m // 2)


    @property
    def embedding_size(self) -> int:
        """Number of output features (possible output states after postselection)."""
        from math import comb
        size = 0
        for k in range(self.postselect, self.n + 1):
            size += comb(self.m, k)
        return size


    def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
        """Delegates the creation of the circuit to the builder."""
        return self.builder.create_circuit(self.m, parameters)

    # TODO... DELTE    
    # def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
    #     """
    #     Create an interferometer circuit with given phase parameters.
    #     If no parameters are provided, uses symbolic placeholders.
    #     """
    #     # (Use Perceval's GenericInterferometer for a mesh of beamsplitters and phase shifters)
    #     if parameters is None:
    #         # Use symbolic parameters if none provided
    #         params = [pcvl.P(f"phi_{i}") for i in range(self.m * (self.m - 1))]
    #         # Duplicate each parameter for paired phase shifters
    #         #params = [val for p in parameters for val in (p, p)]
    #     else:
    #         params = parameters
    #     return pcvl.GenericInterferometer(self.m, lambda idx: (
    #                 pcvl.BS()  # 50:50 beam splitter
    #                 .add(0, pcvl.PS(params[2*idx]))   # phase shifter, then another BS
    #                 .add(0, pcvl.BS())
    #                 .add(0, pcvl.PS(params[2*idx+1]))
    #            ))

    def prepare_processor(self, processor: pcvl.Processor, parameters: list[float]):
        """Configure the given processor with the circuit, input state, and detection filters."""
        processor.set_circuit(self.create_circuit(parameters))
        processor.min_detected_photons_filter(self.postselect)  # enforce minimum detected photons
        #processor.thresholded_output(True)   # <- DEPRECATED # use threshold detectors (no photon-number resolution)
        
        
        from perceval.components import Detector
        for mode in range(self.m):
            processor.add(mode, Detector.threshold())
        
        
        
        # Prepare an input state with n photons evenly spaced in m modes (e.g., [1,0,1,0,...] for 2 photons)
        input_state = [0] * self.m
        if self.n > 0:
            step = max(1, self.m // self.n)
            #step = self.m // self.n
            for i in range(self.n):
                input_state[i * step] = 1
        processor.with_input(pcvl.BasicState(input_state))

    def run(self, parameters: list[float], n_samples: int):
        """
        Run the boson sampler circuit with given phase parameters, sample n_samples times.
        Returns a Perceval BSDistribution (a dict-like object of outcome probabilities).
        """
        if self.session is not None:
            # Build a remote processor via the session (e.g., Scaleway QPU or simulator)
            proc = self.session.build_remote_processor()            # Remote QPU simulation
        else:
            proc = pcvl.Processor(self.backend, self.m)             # Local simulation backend
        # Configure the processor with the circuit and input state
        self.prepare_processor(proc, parameters)
        # Use Perceval's sampling algorithm (Sampler) to run the circuit
        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=n_samples)
        job = sampler.probs(n_samples)  # run sampling
        # If job has the method get_results(), use it; otherwise, assume job is already the result.
        if hasattr(job, "get_results"):
            result = job.get_results() # or job.execute_sync() depending on your workflow
        else:
            result = job

        return result  # BSDistribution of outcomes


    def _distribution_to_feature_vector(self, distribution):
        """
        Convert a distribution dictionary into a fixed-length feature vector.
        """
        feature_vec = torch.zeros(self.embedding_size)
        state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
        for i, state in enumerate(state_list):
            feature_vec[i] = distribution.get(state, 0.0)
        return feature_vec

    def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
        """
        Compute the quantum embedding adaptively.
        Starts with min_samples and doubles until the feature vector converges
        (norm difference < tol) or until max_samples is reached.
        """
        if max_samples is None:
            max_samples = self.n_samples  # default maximum samples from config
        
        prev_embedding = None
        samples = min_samples
        while samples <= max_samples:
            if hasattr(self, "params"):
                distribution = self.run(n_samples=samples)
            else:
                distribution = self.run(parameters=[], n_samples=samples)
            current_embedding = self._distribution_to_feature_vector(distribution)
            if prev_embedding is not None:
                diff = torch.norm(current_embedding - prev_embedding).item()
                if diff < tol:
                    print(f"Adaptive sampling converged with {samples} samples.")
                    return current_embedding
            prev_embedding = current_embedding
            samples *= 2
        print(f"Adaptive sampling reached max samples: {max_samples}")
        return prev_embedding

    # def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
    #     """
    #     Compute the quantum embedding adaptively.
    #     It computes a feature vector (from the output distribution) and doubles the sample count 
    #     until the difference between successive feature vectors is below tol or until max_samples is reached.
    #     """
    #     if max_samples is None:
    #         max_samples = self.n_samples  # default maximum samples from config

    #     # Prepare phase parameters (same as in embed()):
    #     flat = data_tensor.flatten()
    #     if flat.shape[0] > self.nb_parameters:
    #         raise ValueError("Input tensor too large for the number of modes/photons")
    #     phases = torch.zeros(self.nb_parameters)
    #     phases[:flat.shape[0]] = flat
    #     phase_list = (phases * 2 * torch.pi).tolist()

    #     prev_feature = None
    #     samples = min_samples
    #     while samples <= max_samples:
    #         # Run the circuit with the computed phase_list.
    #         distribution = self.run(phase_list, n_samples=samples)
    #         # Convert distribution to a feature vector:
    #         feature_vec = torch.zeros(self.embedding_size)
    #         state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
    #         for i, state in enumerate(state_list):
    #             feature_vec[i] = distribution.get(state, 0.0)
    #         # Compare with the previous feature vector.
    #         if prev_feature is not None:
    #             diff = torch.norm(feature_vec - prev_feature).item()
    #             print(f"Adaptive sampling: {samples} samples, diff = {diff:.6f}")
    #             if diff < tol:
    #                 print(f"Adaptive sampling converged with {samples} samples.")
    #                 return feature_vec
    #         prev_feature = feature_vec
    #         samples *= 2
    #     print(f"Adaptive sampling reached max samples: {max_samples}")
    #     return prev_feature


    # TODO delete
    # def adaptive_embed(self, data_tensor, min_samples=100, max_samples=None, tol=1e-3):
    #     """
    #     Compute the quantum embedding adaptively.
    #     Starts with min_samples and doubles until the embedding converges 
    #     (difference less than tol) or until max_samples is reached.
    #     """
    #     if max_samples is None:
    #         max_samples = self.n_samples  # default maximum samples from config
        
    #     prev_embedding = None
    #     samples = min_samples
    #     while samples <= max_samples:
    #         current_embedding = self.run(parameters=[], n_samples=samples)
    #         if prev_embedding is not None:
    #             diff = torch.norm(current_embedding - prev_embedding).item()
    #             if diff < tol:
    #                 print(f"Adaptive sampling converged with {samples} samples.")
    #                 return current_embedding
    #         prev_embedding = current_embedding
    #         samples *= 2
    #     print(f"Adaptive sampling reached max samples: {max_samples}")
    #     return prev_embedding



    def embed(self, data_tensor, n_samples: int):
        """
        Embed an input tensor (image data) into a quantum feature vector.
        The input tensor values (assumed in [0,1]) are mapped to phase shifts.
        :param data_tensor: input torch tensor with values in [0,1]
        :param n_samples: number of samples for Monte Carlo estimation (ignored if using an analytic mode)
        :return: torch tensor of shape (embedding_size,) with estimated output probabilities.
        """

        # If caching is enabled, try to load the cached embedding.
        if getattr(self, "cache_enabled", False):
            key = hashlib.md5(data_tensor.cpu().numpy().tobytes()).hexdigest()
            cache_file = os.path.join(self.cache_directory, f"{key}.pkl")
            if os.path.exists(cache_file):
                print("Loading embedding from cache.")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)


     # Check if using convolutional embedding.
        from boson_sampler import ConvolutionalInterferometerBuilder  # ensure proper import
        if isinstance(self.builder, ConvolutionalInterferometerBuilder):
            # Convert data_tensor to numpy array (assume shape (1,28,28) or (28,28))
            img_np = data_tensor.squeeze().cpu().numpy()
            phase_list = self.builder.compute_conv_parameters(img_np)
        else:
            # Compute embedding if not cached.
            flat = data_tensor.flatten()
            if flat.shape[0] > self.nb_parameters:
                raise ValueError("Input tensor too large for the number of modes/photons")
            # Prepare phase parameters: pad/truncate to fill the interferometer.
            # Pad or truncate the phases vector to fill all required phase shifters
            #self.nb_parameters == phase_count = self.m * (self.m - 1)  # number of phase parameters needed for full interferometer
            phases = torch.zeros(self.nb_parameters)
            # Use data values (scaled 0-1) for the first part of phases, remaining stay 0
            phases[:flat.shape[0]] = flat
            # Scale phases from [0,1] to [0, 2π] as phase shifts
            phase_list = (phases * 2 * torch.pi).tolist()


        # For variational sampler, self.run expects only n_samples.
        if hasattr(self, "params"):
            distribution = self.run(n_samples=n_samples)
        else:
            distribution = self.run(parameters=phase_list, n_samples=n_samples)
            
        ## Run the circuit and get the output distribution
        ##distribution = self.run(phase_list, n_samples)


        # Convert distribution to a fixed-length probability vector
        #feature_vec = self._distribution_to_feature_vector(distribution)
        feature_vec = torch.zeros(self.embedding_size)
        state_list = self._generate_all_output_states()  # list of BasicState outcomes in canonical order
        for i, state in enumerate(state_list):
            feature_vec[i] = distribution.get(state, 0.0)


        # Save the computed embedding to cache.
        if getattr(self, "cache_enabled", False):
            with open(cache_file, "wb") as f:
                pickle.dump(feature_vec, f)


        return feature_vec

    #@lru_cache(maxsize=1)
    def _generate_all_output_states(self):
        """Precompute all possible output basis states (with postselection).
        Generate and cache all possible output states (as BasicState objects).
        """
        states = []
        for k in range(self.postselect, self.n + 1):
            from itertools import combinations
            for ones in combinations(range(self.m), k):
                bitstring = [1 if j in ones else 0 for j in range(self.m)]
                states.append(pcvl.BasicState(bitstring))
        return states

    from concurrent.futures import ThreadPoolExecutor

    @staticmethod
    def parallel_embed(self, image_list, n_samples, adaptive=False, min_samples=100, tol=1e-3):
        """
        Compute embeddings for a list of images concurrently.
        NOTE: This method is useful for local simulation. 
            However, QaaS (remote) does not handle batch processing well.
            It is recommended to process images sequentially when using the remote platform.
        Parameters:
          image_list : List of image tensors.
          n_samples  : Maximum number of samples for each embedding.
          adaptive   : If True, uses adaptive_embed; otherwise uses embed.
          min_samples: Starting sample count for adaptive sampling.
          tol        : Tolerance for convergence in adaptive sampling.
        Returns:
          A list of embeddings.
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for img in image_list:
                if adaptive:
                    futures.append(executor.submit(self.adaptive_embed, img, min_samples, n_samples, tol))
                else:
                    futures.append(executor.submit(self.embed, img, n_samples))
            embeddings = [f.result() for f in futures]
        return embeddings






class VariationalBosonSampler(BosonSampler):
    def __init__(self, m: int, n: int, postselect: int = None, backend: str = "SLOS", 
                 session=None, builder: InterferometerBuilder = None, cache_enabled=False, 
                 cache_directory="results/cache", init_params=None):
        """
        A variant of BosonSampler where the phase parameters are trainable.
        """
        super().__init__(m, n, postselect, backend, session, builder, cache_enabled, cache_directory)
        # Initialize trainable parameters. If not provided, initialize to small constant values.
        if init_params is None:
            init_params = [0.1] * self.nb_parameters
        # Register parameters as a torch.nn.Parameter so they are updated during training.
        self.params = nn.Parameter(torch.tensor(init_params, dtype=torch.float32))

    adaptive_embed = BosonSampler.adaptive_embed


    def create_circuit(self, parameters: list[float] = None) -> pcvl.Circuit:
        """
        Overrides the create_circuit method to use trainable parameters.
        """
        # Use the current trainable parameters from self.params
        phase_list = self.params.tolist()
        return self.builder.create_circuit(self.m, phase_list)
    
    def run(self, n_samples: int, parameters: list[float] = None):
        """
        Overrides run() to build the circuit using trainable parameters.
        The 'parameters' argument is ignored since self.params is used.
        """
        circuit = self.create_circuit()  # This uses self.params internally.
        if self.session is not None:
            proc = self.session.build_remote_processor()
        else:
            proc = pcvl.Processor(self.backend, self.m)
        # We ignore the passed 'parameters' and use self.params instead.
        self.prepare_processor(proc, parameters=[])  
        sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=n_samples)
        job = sampler.probs(n_samples)
        if hasattr(job, "get_results"):
            result = job.get_results()
        else:
            result = job
        return result




##############
# Enhanced BosonSampler with Caching
# Implementing caching to avoid redundant quantum computations:
############
from functools import lru_cache
import os
import pickle
import hashlib

class CachedBosonSampler(BosonSampler):
    """
    Extended BosonSampler with caching capabilities to avoid redundant computations.
    """
    def __init__(self, m, n, postselect=None, session=None, 
                 cache_enabled=True, cache_size=1000, 
                 disk_cache=False, cache_dir=None):
        super().__init__(m, n, postselect, session)
        self.cache_enabled = cache_enabled
        self.disk_cache = disk_cache
        self.cache_dir = cache_dir
        
        if disk_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Use the wrapper function to apply caching
        if cache_enabled:
            self.embed = self._cached_embed(self.embed, maxsize=cache_size)
    
    def _cached_embed(self, func, maxsize=1000):
        """Apply in-memory caching with LRU strategy."""
        memory_cache = lru_cache(maxsize=maxsize)(func)
        
        def wrapped_func(t, n_sample):
            # Convert tensor to tuple for hashing
            t_tuple = tuple(t.reshape(-1).tolist())
            
            # Check disk cache if enabled
            if self.disk_cache and self.cache_dir:
                cache_key = hashlib.md5(str(t_tuple).encode()).hexdigest()
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
            
            # Use in-memory cache or compute
            result = memory_cache(t, n_sample)
            
            # Save to disk cache if enabled
            if self.disk_cache and self.cache_dir:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                    
            return result
        
        return wrapped_func if self.cache_enabled else func
