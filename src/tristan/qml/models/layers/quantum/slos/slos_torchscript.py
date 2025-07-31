"""
This module extends slos_torch.py with TorchScript-optimized computation graphs
for photonic quantum circuit simulations. It separates the graph construction
from the actual computation for improved performance.

The optimized implementation pre-builds the computation graph based on the input state
configuration, which can then be reused for multiple unitary evaluations.
"""

import math
import torch
import torch.jit as jit

from typing import List, Tuple, Callable, Optional


def prepare_vectorized_operations(operations_list, device=None, dtype=torch.float):
    """
    Convert operations list to tensors for vectorized computation.

    Args:
        operations_list: List of operations, each as [src_idx, dest_idx, mode_i, sqrt_factor]
        device: Optional device to place tensors on (defaults to CPU if None)
        dtype: Data type for the sqrt_factors tensor (default: torch.float)

    Returns:
        Tuple of tensors: (sources, destinations, modes, sqrt_factors)
    """
    if not operations_list:
        return (
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=torch.long, device=device),
            torch.tensor([], dtype=dtype, device=device),
        )

    # Convert operations to tensor format directly on the specified device
    sources = torch.tensor(
        [op[0] for op in operations_list], dtype=torch.long, device=device
    )
    destinations = torch.tensor(
        [op[1] for op in operations_list], dtype=torch.long, device=device
    )
    modes = torch.tensor(
        [op[2] for op in operations_list], dtype=torch.long, device=device
    )
    sqrt_factors = torch.tensor(
        [op[3] for op in operations_list], dtype=dtype, device=device
    )

    return sources, destinations, modes, sqrt_factors


def layer_compute_vectorized(
    unitary: torch.Tensor,
    prev_amplitudes: torch.Tensor,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    modes: torch.Tensor,
    sqrt_factors: torch.Tensor,
    p: int,
) -> torch.Tensor:
    """
    Compute amplitudes for a single layer using vectorized operations.

    Args:
        unitary: Batch of unitary matrices [batch_size, m, m]
        prev_amplitudes: Previous layer amplitudes [batch_size, prev_size]
        sources: Source indices for operations [num_ops]
        destinations: Destination indices for operations [num_ops]
        modes: Mode indices for operations [num_ops]
        sqrt_factors: Square root factors for operations [num_ops]
        p: Photon index for this layer

    Returns:
        Next layer amplitudes [batch_size, next_size]
    """
    batch_size = unitary.shape[0]

    # Handle empty operations case
    if sources.shape[0] == 0:
        return prev_amplitudes

    # Determine output size
    next_size = int(destinations.max().item()) + 1

    # Create result tensor with same dtype as input
    result = torch.zeros(
        (batch_size, next_size), dtype=prev_amplitudes.dtype, device=unitary.device
    )

    # Get unitary elements for all operations
    # Shape: [batch_size, num_ops]
    u_elements = unitary[:, modes, p]

    # Get source amplitudes for all operations
    # Shape: [batch_size, num_ops]
    prev_amps = prev_amplitudes[:, sources]

    # Compute contributions
    # Shape: [batch_size, num_ops]
    contributions = u_elements * prev_amps * sqrt_factors

    # Use index_add_ for each batch item to accumulate results
    for b in range(batch_size):
        result[b].index_add_(0, destinations, contributions[b])

    return result


class SLOSComputeGraph:
    """
    A class that builds and stores the computation graph for SLOS algorithm.

    This separates the graph construction (which depends only on input state, no_bunching,
    and output_map_func) from the actual computation using the unitary matrix.
    """

    def __init__(
        self,
        m: int,
        input_state: List[int],
        output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
        no_bunching: bool = False,
        keep_keys: bool = True,
        device=None,  # Optional device parameter
        dtype: torch.dtype = torch.float,  # Optional dtype parameter
    ):
        """
        Initialize the SLOS computation graph.

        Args:
            m (int): Number of modes in the circuit
            input_state (List[int]): List of integers specifying number of photons in each input mode
            output_map_func (callable, optional): Function that maps output states
            no_bunching (bool): If True, the algorithm is optimized for no-bunching states only
            keep_keys (bool): If True, output state keys are returned
            device: Optional device to place tensors on (CPU, CUDA, etc.)
            dtype: Data type precision for floating point calculations (default: torch.float)
                  Use torch.float16 for half precision, torch.float for single precision,
                  or torch.float64 for double precision
        """
        self.m = m
        self.input_state = input_state
        self.output_map_func = output_map_func
        self.no_bunching = no_bunching
        self.keep_keys = keep_keys
        self.device = device
        self.dtype = dtype

        # Determine corresponding complex dtype
        if dtype == torch.float16:
            self.complex_dtype = torch.complex32
        elif dtype == torch.float:
            self.complex_dtype = torch.cfloat
        elif dtype == torch.float64:
            self.complex_dtype = torch.cdouble
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype}. Must be torch.float16, torch.float, or torch.float64"
            )

        # Check input validity
        if any(n < 0 for n in input_state) or sum(input_state) == 0:
            raise ValueError("Photon numbers cannot be negative or all zeros")

        if no_bunching and not all(x in [0, 1] for x in input_state):
            raise ValueError(
                "Input state must be binary (0s and 1s only) in non-bunching mode"
            )

        # Build computation graph structure
        self.n_photons = sum(input_state)
        self.idx_n = []  # Input mode indices for each photon
        for i, count in enumerate(input_state):
            for _ in range(count):
                self.idx_n.append(i)

        # Pre-compute layer structures and operation sequences
        self._build_graph_structure()

        # Create TorchScript function for the core computation
        self._create_torchscript_modules()

    def _build_graph_structure(self):
        """Build the graph structure using dictionary for fast state lookups."""
        self.layer_sizes = []  # Size of each layer's amplitude tensor
        list_operations = []  # Operations to perform at each layer
        self.vectorized_operations = []  # the same, vectorized

        # Initial state is all zeros
        last_combinations = {tuple([0] * self.m): 0}
        self.layer_sizes.append(1)

        input_state_tensor = torch.zeros(self.m, dtype=self.dtype, device=self.device)

        # For each photon/layer, compute the state combinations and operations
        for idx, p in enumerate(self.idx_n):
            input_state_tensor[p] += 1

            # Calculate number of combinations for this layer
            layer_size = (
                self.no_bunching
                and math.comb(self.m, idx + 1)
                or math.comb(self.m + idx, idx + 1)
            )
            self.layer_sizes.append(layer_size)

            combinations = {}
            operations = []  # [src_state_idx, dest_idx, mode_i, sqrt_factor]

            for state, src_state_idx in last_combinations.items():
                nstate = list(state)
                for i in range(self.m):
                    if nstate[i] and self.no_bunching:
                        continue

                    nstate[i] += 1
                    nstate_tuple = tuple(nstate)

                    dest_idx = combinations.get(nstate_tuple, None)
                    if dest_idx is None:
                        dest_idx = combinations[nstate_tuple] = len(combinations)

                    sqrt_factor = math.sqrt(
                        nstate[i] / float(input_state_tensor[p].item())
                    )

                    # Record the operation: [src_state_idx, dest_idx, mode_i, sqrt_factor]
                    operations.append([src_state_idx, dest_idx, i, sqrt_factor])

                    nstate[i] -= 1

            list_operations.append(operations)
            last_combinations = combinations

        # For each layer, prepare vectorized operations on the specified device
        for ops in list_operations:
            sources, destinations, modes, sqrt_factors = prepare_vectorized_operations(
                ops, device=self.device
            )
            self.vectorized_operations.append(
                (sources, destinations, modes, sqrt_factors)
            )

        # Store only the final layer combinations if needed for output mapping or keys
        self.final_keys = (
            list(last_combinations.keys())
            if self.keep_keys or self.output_map_func
            else None
        )
        del last_combinations

        if self.output_map_func is not None:
            self.mapped_keys = []
            mapping_indices = {}  # Maps mapped state to its index
            self.mapped_indices = []  # For each original state, store the mapped index

            for idx, key in enumerate(self.final_keys):
                mapped_state = self.output_map_func(key)
                if mapped_state is not None:
                    if mapped_state not in mapping_indices:
                        mapping_indices[mapped_state] = len(self.mapped_keys)
                        self.mapped_keys.append(mapped_state)

                    mapped_idx = mapping_indices[mapped_state]
                    self.mapped_indices.append(mapped_idx)
                else:
                    # Discarded state
                    self.mapped_indices.append(-1)

            self.total_mapped_keys = len(self.mapped_keys)

            # Clean up temporary dictionaries
            del mapping_indices
        else:
            self.mapped_keys = self.final_keys
            self.total_mapped_keys = self.keep_keys and len(self.final_keys) or 0

    def _create_torchscript_modules(self):
        """Create TorchScript modules for different parts of the computation."""
        # Create layer computation functions
        self.layer_functions = []

        for layer_idx, (sources, destinations, modes, sqrt_factors) in enumerate(
            self.vectorized_operations
        ):
            # Get the photon index for this layer
            p = self.idx_n[layer_idx]

            # Create a partial function with fixed operations and photon index
            def make_layer_fn(s, d, m, f, p_val):
                return lambda u, prev: layer_compute_vectorized(
                    u, prev, s, d, m, f, p_val
                )

            self.layer_functions.append(
                make_layer_fn(sources, destinations, modes, sqrt_factors, p)
            )

        # Create mapping function if needed
        if self.output_map_func is not None:

            @jit.script
            def apply_mapping(
                probabilities: torch.Tensor, mapping: List[int], output_size: int
            ) -> torch.Tensor:
                """Apply state mapping and accumulate probabilities."""
                batch_size = probabilities.shape[0]
                # Create result tensor on the same device as input
                result = torch.zeros(
                    (batch_size, output_size),
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )

                for idx, mapped_idx in enumerate(mapping):
                    if mapped_idx >= 0:  # Not discarded
                        result[:, mapped_idx] += probabilities[:, idx]

                # Renormalize - simpler approach
                sum_probs = result.sum(dim=1, keepdim=True)
                # Avoid division by zero
                safe_sum = torch.where(
                    sum_probs > 0, sum_probs, torch.ones_like(sum_probs)
                )

                # Normalize all batches
                normalized_result = result / safe_sum
                return normalized_result

            self.mapping_function = lambda probs: apply_mapping(
                probs, self.mapped_indices, self.total_mapped_keys
            )
        else:
            self.mapping_function = lambda x: x

    def compute(
        self, unitary: torch.Tensor
    ) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
        """
        Compute the probability distribution using the pre-built graph.

        Args:
            unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m]
                The unitary should be provided in the complex dtype corresponding to the graph's dtype.
                For example, for torch.float32, use torch.cfloat; for torch.float64, use torch.cdouble.

        Returns:
            Tuple[List[Tuple[int, ...]], torch.Tensor]:
                - List of tuples representing output Fock state configurations
                - Probability distribution tensor
        """
        if len(unitary.shape) == 2:
            is_batched = False
            unitary = unitary.unsqueeze(0)  # Add batch dimension [1 x m x m]
        else:
            is_batched = True

        batch_size, m, m2 = unitary.shape
        if m != m2 or m != self.m:
            raise ValueError(
                f"Unitary matrix must be square with dimension {self.m}x{self.m}"
            )

        # Check and convert dtype if necessary
        if unitary.dtype != self.complex_dtype:
            # Get common type based on precision needs
            target_type = self.complex_dtype
            if self.dtype == torch.float16 and not hasattr(torch, "complex32"):
                # Fall back to cfloat for platforms without complex32
                target_type = torch.cfloat
                print(
                    f"Warning: torch.complex32 not available, using {target_type} instead"
                )

            unitary = unitary.to(dtype=target_type)
            print(f"Warning: Converted unitary from {unitary.dtype} to {target_type}")

        # Get device from unitary
        device = unitary.device

        # Initial amplitude (batch of 1s on same device as unitary with appropriate dtype)
        amplitudes = torch.ones(
            (batch_size, 1), dtype=self.complex_dtype, device=device
        )

        # Apply each layer
        for layer_idx, layer_fn in enumerate(self.layer_functions):
            amplitudes = layer_fn(unitary, amplitudes)

        # Calculate probabilities
        probabilities = (amplitudes.abs() ** 2).real

        # Apply output mapping if needed
        if self.output_map_func is not None:
            probabilities = self.mapping_function(probabilities)
            keys = self.mapped_keys
        else:
            if self.no_bunching:
                sum_probs = probabilities.sum(dim=1, keepdim=True)
                # Only normalize when sum > 0 to avoid division by zero
                valid_entries = sum_probs > 0
                if valid_entries.any():
                    probabilities = torch.where(
                        valid_entries,
                        probabilities
                        / torch.where(
                            valid_entries, sum_probs, torch.ones_like(sum_probs)
                        ),
                        probabilities,
                    )
            keys = self.final_keys if self.keep_keys else None

        # Remove batch dimension if input was single unitary
        if not is_batched:
            probabilities = probabilities.squeeze(0)

        return keys, probabilities


def build_slos_compute_graph(
    input_state: List[int],
    output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
    no_bunching: bool = False,
    keep_keys: bool = True,
    device=None,
    dtype: torch.dtype = torch.float,
) -> SLOSComputeGraph:
    """
    Build a computation graph for SLOS algorithm that can be reused for multiple unitaries.

    The graph is constructed based on the input state configuration and can be reused
    for multiple evaluations with different unitaries, providing significant performance
    benefits for repeated calculations.

    Args:
        input_state (List[int]): List of integers specifying number of photons in each input mode.
            For example, [1, 1, 0] means one photon in each of the first two modes.

        output_map_func (callable, optional): Function that maps output states to new states
            or None. If the function returns None for a state, that state is discarded.
            Example mapping functions include threshold_mapping for binary detection.
            Default is None (no mapping).

        no_bunching (bool): If True, the algorithm is optimized for no-bunching states only.
            This is appropriate for many experiments where each input photon must be
            detected exactly once in the output. Default is False.

        keep_keys (bool): If True, the output state keys are returned. If you only need
            the probabilities and not the state configurations, set to False for memory
            efficiency. Default is True.

        device: Optional device to place tensors on (CPU, CUDA, MPS, etc.). If None,
            tensors will be created on CPU and moved to the appropriate device during
            computation based on the unitary's device. Specifying a device can improve
            performance by avoiding device transfers.

        dtype (torch.dtype): Data type precision for floating point calculations.
            Options:
            - torch.float16 (half precision): Fastest but least accurate
            - torch.float (single precision): Good balance of speed and accuracy (default)
            - torch.float64 (double precision): Highest accuracy but slowest
            The unitary matrix should use the corresponding complex dtype
            (complex32, cfloat, or cdouble) for optimal performance.

    Returns:
        SLOSComputeGraph: Computation graph object that can be used to compute
        output probability distributions for multiple unitaries with the same
        input configuration.

    Examples:
        >>> import torch
        >>> from pcvl_pytorch import build_slos_compute_graph
        >>>
        >>> # Create a computation graph for Hong-Ou-Mandel input
        >>> input_state = [1, 1]
        >>> graph = build_slos_compute_graph(input_state)
        >>>
        >>> # Use the graph with different unitaries
        >>> unitary1 = torch.tensor([[0.7071, 0.7071], [0.7071, -0.7071]], dtype=torch.cfloat)
        >>> keys, probs1 = graph.compute(unitary1)
        >>>
        >>> # High precision calculation
        >>> graph_hp = build_slos_compute_graph(input_state, dtype=torch.float64)
        >>> unitary2 = torch.tensor([[0.7071, 0.7071], [0.7071, -0.7071]], dtype=torch.cdouble)
        >>> keys, probs2 = graph_hp.compute(unitary2)
    """

    m = len(input_state)
    return SLOSComputeGraph(
        m, input_state, output_map_func, no_bunching, keep_keys, device, dtype
    )


def pytorch_slos_output_distribution(
    unitary: torch.Tensor,
    input_state: List[int],
    output_map_func: Callable[[Tuple[int, ...]], Optional[Tuple[int, ...]]] = None,
    no_bunching: bool = False,
    keep_keys: bool = True,
) -> Tuple[List[Tuple[int, ...]], torch.Tensor]:
    """
    TorchScript-optimized version of pytorch_slos_output_distribution.

    This function builds the computation graph first, then uses it to compute the probabilities.
    For repeated calculations with the same input configuration but different unitaries,
    it's more efficient to use build_slos_compute_graph() directly.

    Args:
        unitary (torch.Tensor): Single unitary matrix [m x m] or batch of unitaries [b x m x m]
        input_state (List[int]): List of integers specifying number of photons in each input mode
        output_map_func (callable, optional): Function that maps output states
        no_bunching (bool): If True, the algorithm is optimized for no-bunching states only
        keep_keys (bool): If True, output state keys are returned

    Returns:
        Tuple[List[Tuple[int, ...]], torch.Tensor]:
            - List of tuples representing output Fock state configurations
            - Probability distribution tensor
    """
    # Extract device from unitary for graph building
    device = unitary.device if hasattr(unitary, "device") else None

    # Determine appropriate dtype based on unitary's complex dtype
    if unitary.dtype == torch.complex32:
        dtype = torch.float16
    elif unitary.dtype == torch.cfloat:
        dtype = torch.float
    elif unitary.dtype == torch.cdouble:
        dtype = torch.float64
    else:
        dtype = torch.float  # Default to float32 for unknown types

    # Build graph on the same device as the unitary with matching precision
    graph = build_slos_compute_graph(
        input_state, output_map_func, no_bunching, keep_keys, device=device, dtype=dtype
    )
    return graph.compute(unitary)
